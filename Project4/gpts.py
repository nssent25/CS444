'''gpts.py
Family of Generated Pretrained Transformer (GPT) neural networks.
Nithun Selva and Saad Khan
CS 444: Deep Learning
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import network
from tf_util import arange_index
from layers import Dense
from transformer_layers import Embedding
from transformer_blocks import TransformerBlock, PositionalEncodingBlock


class GPT(network.DeepNetwork):
    '''The parent class of all GPT networks. Customizes the functionality of DeepNetwork for transformers.
    '''
    def __init__(self, seq_len, padding_char_enc, reg=0):
        '''GPT constructor

        Parameters:
        -----------
        seq_len: int.
            The length of sequences processed by the transformer (`T`).
        padding_char_enc: int.
            The INT-CODED padding char. Needed to compute the temporal cross-entropy loss.
        reg: float.
            Regularization strength.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create instance variables for the method parameters as needed.
        '''
        super().__init__(input_feats_shape=(seq_len,), reg=reg)
        self.seq_len = seq_len
        self.padding_char_enc = padding_char_enc
        self.layers = []  # List of layers in the network

    def get_one_fake_input(self):
        '''Generates a fake input for use in the `compile` method to trigger lazy initialization to initialize network
        parameters.

        This function is provided to you. You should not need to modify it.
        '''
        return tf.zeros(shape=(1, *self.input_feats_shape), dtype=tf.int32)

    def loss(self, out_net_act, y, eps=1e-16, mask_padding_preds=True):
        '''Computes the loss for the current minibatch based on the output layer activations `out_net_act` and int-coded
        class labels `y`.

        This function is provided to you. You should not need to modify it.

        Parameters:
        -----------
        output_layer_net_act: tf.constant. shape=(B, T, C) or None.
            Net activation in the output layer for the current mini-batch.
        y: tf.constant. shape=(B,). tf.int32s.
            int-coded true classes for the current mini-batch.
        eps: float.
            Small fudge factor to use when evaluating the cross entropy log to prevent taking log of 0.
        mask_padding_preds: bool.
            Should we not count time steps in mini-batch sequences when the current character is the padding character
            and the prediction would then be meaningless?1

        Returns:
        -----------
        float.
            The loss.

        NOTE:
        - We treat the time dimension as additional data samples from the perspective of the loss function.
        - When we are not masking out padding chars from the temporal cross entropy loss, the loss simplifies to
        regular cross entropy loss.
        '''
        N, T, vocab_sz = out_net_act.shape

        if self.loss_name == 'cross_entropy':
            act_at_correct = arange_index(out_net_act, y)
            loss = tf.reduce_mean(-tf.math.log(act_at_correct + eps)) # sparse int based
        elif self.loss_name == 'temporal_cross_entropy': # NEW TRANSFORMERS
            # Our input is 3D so we need to combine/treat time as more batches.
            out_net_act_flat = tf.reshape(out_net_act, [N*T, vocab_sz])
            y_flat = tf.reshape(y, (N*T,))

            # Compute the per-sample loss like usual
            act_at_correct = arange_index(out_net_act_flat, y_flat)
            print(tf.reshape(act_at_correct, (N, T)))
            loss = -tf.math.log(act_at_correct + eps)

            if mask_padding_preds:
                # We want to nix padding chars from the loss, so mask them out
                mask = tf.cast(y_flat != self.padding_char_enc, dtype=tf.float32)
                N_effective = tf.reduce_sum(mask)  # We want to avg loss by number of real non-pad contributions
                loss = mask * loss  # (N*T,) but only non-padded chars contribute
                loss = (1/N_effective) * tf.reduce_sum(loss) # sparse int based
            else:
                # Average all per-sample loss values together
                loss = tf.reduce_mean(loss)
        else:
            raise ValueError(f'Unknown loss function {self.loss_name}')

        return loss

    def __call__(self, x):
        '''Forward pass through the Transformer with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, M).
            B data samples (seqs) of length T. Each time step has `M`=vocab_sz features.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, C).
            Activations produced by the output layer to the mini-batch for all time steps and possible tokens in the
            vocab.

        Hint: All layers/blocks in all GPTs are connected sequentially at the macro network level...;)
        '''
        for layer in self.layers:
            x = layer(x)
        return x

    def generate_sequence(self, prompt, length, char2ind_map, ind2char_map, end_char=None, method='max',
                          plot_probs=False, live_print=True, line_width=80):
        '''Generates/predicts a sequence of chars of length `length` chars that follow the provided prompt.
        It is helpful remember that the transformer generates chars one at a time sequentially. Therefore in
        prediction/generation mode, the network processes tokens in mini-batches of one item for one time step.

        Parameters:
        -----------
        prompt: str or list.
            Chars/tokens that the transformer should process and predict tokens after.
        length: int.
            Number of tokens that the transformer generates after the prompt.
        char2ind_map: Python dictionary.
            Keys: chars/tokens in vocab. Values: int code in the vocab.
        ind2char_map: Python dictionary.
            Keys: int code in the vocab. Values: corresponding char/token.
        end_char: str or None.
            If specified, generation stops when this character is produced.
        method: str.
            'max' for deterministic generation or 'distributed' for sampling.
        plot_probs: bool.
            Whether to plot probabilities during generation.
        live_print: bool.
            Whether to print generated text in real-time.
        line_width: int.
            Maximum width of a line before wrapping (only affects display).

        Returns:
        -----------
        list.
            The sequence of generated tokens (without the prompt).
        '''
        # Turn off Dropout
        self.set_layer_training_mode(is_training=False)

        # Convert prompt to list of integer tokens
        if isinstance(prompt, str):
            prompt_ints = [char2ind_map[char] for char in prompt]
        else:
            prompt_ints = prompt  # Assume already tokenized

        # response (int coded) we will return
        response_int = []

        # RNG for distributed multinomial sampling
        rng = np.random.default_rng()

        # print prompt and track current line length
        current_line_length = 0
        if live_print:
            print(prompt, sep='', end='')
            current_line_length = len(prompt)

        # Where we get the next char prediction from the output layer net_acts
        t_ind = len(prompt_ints)-1

        # Second: now that state has developed, generate new chars using a feedback loop (prev pred = next input)
        for t in range(length):
            # Now we assemble the context that is fed to the transformer. This is the user's prompt + what has been
            # generated
            context = prompt_ints + response_int

            # Ensure we keep the context length to the T used to train the transformer. Use the T most recent tokens as
            # a context (rolling window)
            if len(context) > self.seq_len:
                context = context[-self.seq_len:]

            # Make sure we left-pad so that the context input we give the transformer is always seq_len long
            # and the prompt is next to the generated text
            while len(context) < self.seq_len:
                context.append(self.padding_char_enc)

            # Convert int into Tensor format (1, T)
            x_input_tf = tf.expand_dims(context, axis=0)  # N, T
            # Get net_acts from output layer
            out_net_act = self(x_input_tf)

            # Important: Squeeze / convert to numpy before drawing from softmax dist.
            out_probs_np = out_net_act[0, t_ind].numpy()  # probs from only last time step
            # Draw predicted char index from vocab proportional to the softmax prob
            if method == 'max':
                curr_pred_int = tf.argmax(out_net_act[0, t_ind, :]).numpy()
            else:
                curr_pred_int = int(rng.choice(np.arange(len(out_probs_np)), p=out_probs_np))

            if plot_probs:
                plt.plot(out_probs_np)
                plt.title(f'{t}th char: pred char: {ind2char_map[curr_pred_int]} ({curr_pred_int})')
                plt.show()

            # Append to generated seq, the char corresponding to the predicted index in the vocab
            response_int.append(curr_pred_int)

            # Advance where we get our prediction next time.
            if t_ind < self.seq_len - 1 and t_ind > 0:
                t_ind += 1
            else:
                t_ind = -1

            # Print next char live as we go with word wrap
            if live_print:
                curr_char = ind2char_map[curr_pred_int]
                print(curr_char, sep='', end='')
                current_line_length += 1

                # Check if we need to wrap
                if current_line_length >= line_width:
                    if curr_char in [' ', '\n', '.', ',', ':', ';', '!', '?']:
                        print()  # Insert a line break
                        current_line_length = 0

                # If we encounter a natural line break, reset counter
                if curr_char == '\n':
                    current_line_length = 0

            if end_char is not None and curr_pred_int == char2ind_map[end_char]:
                break

        if live_print:
            print()

        seq_char = [ind2char_map[index] for index in response_int]

        if plot_probs:
            plt.show()

        return seq_char


class GPTPico1(GPT):
    '''GPTPico1 is a "baby GPT" — the shallowest transformer in the family. It consists of only one transformer block.

    Here is a summary the architecture:

    Embedding layer → Position Encoding block → Transformer block (1x) → Dense output layer

    In the output layer:
    - Use He init.
    - Use layer norm.
    '''
    def __init__(self, vocab_sz, seq_len, padding_char_enc=None, num_heads=4, embed_dim=24, dropout_rate=0.0):
        '''GPTPico1 constructor

        Parameters:
        -----------
        vocab_sz: int.
            The vocab size (`M`).
        seq_len: int.
            Length of the sequences, in time steps, that the transformer processes (`T`)
        padding_char_enc: int.
            The INT-CODED padding char. Needed to compute the temporal cross-entropy loss.
        num_heads: int.
            Number of attention heads to use in the attention blocks.
        embed_dim: int.
            The number of units in the Embedding layer.
        dropout_rate: float.
            The dropout rate (R) to use throughout the net.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(seq_len=seq_len, padding_char_enc=padding_char_enc)
        self.vocab_sz = vocab_sz
        self.embed_dim = embed_dim

        self.embedding_layer = Embedding(name='EmbeddingLayer',
                                         input_dim=vocab_sz,
                                         embed_dim=embed_dim,
                                         prev_layer_or_block=None)
        self.layers.append(self.embedding_layer)  

        self.pos_encoding_block = PositionalEncodingBlock(blockname='PositionalEncodingBlock',
                                                          embed_dim=embed_dim,
                                                          prev_layer_or_block=self.embedding_layer,
                                                          dropout_rate=dropout_rate)
        self.layers.append(self.pos_encoding_block)

        transformer_block_0 = TransformerBlock(blockname='TransformerBlock_0',
                                               units=embed_dim,
                                               num_heads=num_heads,
                                               prev_layer_or_block=self.pos_encoding_block,
                                               dropout_rate=dropout_rate)
        self.layers.append(transformer_block_0)

        self.output_layer = Dense(name='output',
                                        units=vocab_sz,
                                        activation='softmax', 
                                        wt_init='he',
                                        do_layer_norm=True,
                                        prev_layer_or_block=transformer_block_0)
        self.layers.append(self.output_layer)  


class GPTMini6(GPT):
    '''GPTMini6 is a moderate-sized GPT in the family. It consists of 6 transformer blocks.

    Here is a summary the architecture:

    Embedding layer → Position Encoding block → Transformer block (6x) → Dense output layer

    In the output layer:
    - Use He init.
    - Use layer norm.
    '''
    def __init__(self, vocab_sz, seq_len, padding_char_enc=None, num_heads=6, embed_dim=384, dropout_rate=0.2):
        '''GPTMini6 constructor

        Parameters:
        -----------
        vocab_sz: int.
            The vocab size (`M`).
        seq_len: int.
            Length of the sequences, in time steps, that the transformer processes (`T`)
        padding_char_enc: int.
            The INT-CODED padding char. Needed to compute the temporal cross-entropy loss.
        num_heads: int.
            Number of attention heads to use in the attention blocks.
        embed_dim: int.
            The number of units in the Embedding layer.
        dropout_rate: float.
            The dropout rate (R) to use throughout the net.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(seq_len=seq_len, padding_char_enc=padding_char_enc)
        self.vocab_sz = vocab_sz
        self.embed_dim = embed_dim

        self.embedding_layer = Embedding(name='EmbeddingLayer',
                                         input_dim=vocab_sz,
                                         embed_dim=embed_dim,
                                         prev_layer_or_block=None)
        self.layers.append(self.embedding_layer)

        self.pos_encoding_block = PositionalEncodingBlock(blockname='PositionalEncodingBlock',
                                                          embed_dim=embed_dim,
                                                          prev_layer_or_block=self.embedding_layer,
                                                          dropout_rate=dropout_rate)
        self.layers.append(self.pos_encoding_block)

        prev_block_for_transformer = self.pos_encoding_block
        for i in range(6): # 6 Transformer blocks
            tb = TransformerBlock(blockname=f'TransformerBlock_{i}',
                                  units=embed_dim,
                                  num_heads=num_heads,
                                  prev_layer_or_block=prev_block_for_transformer,
                                  dropout_rate=dropout_rate)
            self.layers.append(tb)
            prev_block_for_transformer = tb

        self.output_layer = Dense(name='output',
                                        units=vocab_sz,
                                        activation='softmax',
                                        wt_init='he',
                                        do_layer_norm=True,
                                        prev_layer_or_block=prev_block_for_transformer)
        self.layers.append(self.output_layer)


class GPT1(GPT):
    '''Implementation of OpenAI's GPT-1 architecture
    
    Architecture:
    - 12 stacked Transformer Blocks
    - 12 attention heads
    - Embedding dimension of 768
    - Dropout rate of 0.2
    '''
    def __init__(self, vocab_sz, seq_len, padding_char_enc=None):
        '''GPT1 constructor'''
        super().__init__(seq_len=seq_len, padding_char_enc=padding_char_enc)
        self.vocab_sz = vocab_sz
        
        # GPT-1 specific parameters
        num_heads = 12
        embed_dim = 768
        dropout_rate = 0.2
        
        # Embedding layer
        self.embedding_layer = Embedding(name='EmbeddingLayer',
                                         input_dim=vocab_sz,
                                         embed_dim=embed_dim,
                                         prev_layer_or_block=None)
        self.layers.append(self.embedding_layer)
        
        # Positional encoding
        self.pos_encoding_block = PositionalEncodingBlock(
            blockname='PositionalEncodingBlock',
            embed_dim=embed_dim,
            prev_layer_or_block=self.embedding_layer,
            dropout_rate=dropout_rate
        )
        self.layers.append(self.pos_encoding_block)
        
        # 12 Transformer blocks
        prev_block_for_transformer = self.pos_encoding_block
        for i in range(12):
            tb = TransformerBlock(
                blockname=f'TransformerBlock_{i}',
                units=embed_dim,
                num_heads=num_heads,
                prev_layer_or_block=prev_block_for_transformer,
                dropout_rate=dropout_rate
            )
            self.layers.append(tb)
            prev_block_for_transformer = tb
        
        # Output layer
        self.output_layer = Dense(
            name='output',
            units=vocab_sz,
            activation='softmax',
            wt_init='he',
            do_layer_norm=True,
            prev_layer_or_block=prev_block_for_transformer
        )
        self.layers.append(self.output_layer)


class GPT2XL(GPT):
    '''GPT2XL is an implementation of the GPT-2 XL architecture.
    
    Architecture:
    - 48 transformer blocks
    - 25 attention heads
    - Embedding dimension of 1600
    - Dropout rate of 0.1

    All parameters are customizable in the constructor.
    '''
    def __init__(self, vocab_sz, seq_len, padding_token_enc=None, num_heads=25, 
                 embed_dim=1600, n_layers=48, dropout_rate=0.1):
        '''GPT2XL constructor
        
        Parameters:
        -----------
        vocab_sz: int.
            The vocab size from BPE tokenization
        seq_len: int.
            Length of the sequences the transformer processes
        padding_token_enc: int.
            The INT-CODED padding token
        num_heads: int.
            Number of attention heads (25 for GPT-2 XL)
        embed_dim: int.
            Embedding dimension (1600 for GPT-2 XL)
        n_layers: int.
            Number of transformer layers (48 for GPT-2 XL)
        dropout_rate: float.
            Dropout rate throughout the network
        '''
        super().__init__(seq_len=seq_len, padding_char_enc=padding_token_enc)
        self.vocab_sz = vocab_sz
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding_layer = Embedding(name='EmbeddingLayer',
                                         input_dim=vocab_sz,
                                         embed_dim=embed_dim,
                                         prev_layer_or_block=None)
        self.layers.append(self.embedding_layer)
        
        # Positional encoding
        self.pos_encoding_block = PositionalEncodingBlock(blockname='PositionalEncodingBlock',
                                                          embed_dim=embed_dim,
                                                          prev_layer_or_block=self.embedding_layer,
                                                          dropout_rate=dropout_rate)
        self.layers.append(self.pos_encoding_block)
        
        # Transformer blocks (48 for XL)
        prev_block_for_transformer = self.pos_encoding_block
        for i in range(n_layers):
            tb = TransformerBlock(blockname=f'TransformerBlock_{i}',
                                  units=embed_dim,
                                  num_heads=num_heads,
                                  prev_layer_or_block=prev_block_for_transformer,
                                  dropout_rate=dropout_rate)
            self.layers.append(tb)
            prev_block_for_transformer = tb
        
        # Output layer
        self.output_layer = Dense(name='output',
                                 units=vocab_sz,
                                 activation='softmax',
                                 wt_init='he',
                                 do_layer_norm=True,
                                 prev_layer_or_block=prev_block_for_transformer)
        self.layers.append(self.output_layer)