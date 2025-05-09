'''transformer_blocks.py
Blocks related to transformer neural networks.
YOUR NAMES HERE
CS 444: Deep Learning
'''
import tensorflow as tf

import block
from layers import Dense, Dropout
from transformer_layers import PositionalEncoding
from tf_util import tril


class QueryKeyValueBlock(block.Block):
    '''Block that encapsulates the Dense layers that generate the queries, keys, and values.'''
    def __init__(self, blockname, units, prev_layer_or_block):
        '''QueryKeyValueBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units. int.
            The number of neurons in each of the Dense layers in the block. All Dense layers have the same number of
            units (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        Properties of all layers:
        ---------------------------
        - They are along separate branches. Think about what this means for their previous layer/block reference.
        - He initialization.
        - Layer normalization.
        - Linear/identity activation.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Assemble layers in the block.
        '''
        # 1) init the block itself
        super().__init__(blockname, prev_layer_or_block)

        # 2) create the three parallel Dense layers
        #    all with linear act, He init, and layer‐norm ON
        self.query = Dense(
            name=f"{blockname}_Q",
            units=units,
            activation='linear',
            wt_init='he',
            prev_layer_or_block=prev_layer_or_block,
            do_layer_norm=True
        )
        self.key   = Dense(
            name=f"{blockname}_K",
            units=units,
            activation='linear',
            wt_init='he',
            prev_layer_or_block=prev_layer_or_block,
            do_layer_norm=True
        )
        self.value = Dense(
            name=f"{blockname}_V",
            units=units,
            activation='linear',
            wt_init='he',
            prev_layer_or_block=prev_layer_or_block,
            do_layer_norm=True
        )

        # 3) register them so the block can toggle train/eval mode, etc.
        self.layers = [self.query, self.key, self.value]


    def __call__(self, query_input, key_input, value_input):
        '''Forward pass through the QKV Block with activations that should represent the input to respective QKV layers.

        Parameters:
        -----------
        query_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the query layer.
        key_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the key layer.
        value_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the value layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the query layer.
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the key layer.
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the value layer.
        '''
        # each branch goes into its matching Dense
        q = self.query(query_input)
        k = self.key(key_input)
        v = self.value(value_input)

        return q, k, v


class AttentionBlock(block.Block):
    '''Block that encapsulates the fundamental attention mechanism.'''
    def __init__(self, blockname, num_heads, units, prev_layer_or_block, dropout_rate=0.1, causal=True):
        '''AttentionBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        num_heads: int.
            Number of attention heads to use in the attention block.
        units: int.
            Number of neurons in the attention block (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer that is in the attention block that is applied to the
            attention values.
        causal: bool.
            Whether to apply a causal mask to remove/mask out the ability for the layer to pay attention to tokens
            that are in the future of the current one in the sequence.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create any instance variables to save any information that will be helpful to access during the forward pass.
        3. Create the Dropout layer.
        4. For efficiency, it is helpful to pre-compute the attention gain and assign it to an instance variable
        (e.g. as self.gain) so that you can use it during the forward pass. You have all the info here that is needed
        to compute the gain.

        NOTE: Remember to add your dropout layer to the layers list (otherwise the dropout mode will not get set) and
        to make the dropout layer's prev reference whatever is passed into this block.
        '''
        super().__init__(blockname, prev_layer_or_block)

        self.num_heads = num_heads
        self.units     = units      # H_qkv
        self.causal    = causal

        # precompute the attention gain: 1 / sqrt(H_qkv / A)
        self.gain = 1.0 / tf.sqrt(tf.cast(units  / num_heads, tf.float32))

        # the dropout over attention weight matrix A_3 → A_4
        self.attn_dropout = Dropout(
            name=f"{blockname}_dropout",
            rate=dropout_rate,
            prev_layer_or_block=self
        )

        # register any sub-layers so mode toggles etc. work
        self.layers = [ self.attn_dropout ]

    def __call__(self, queries, keys, values):
        '''Forward pass through the attention block with activations from the query, key, and value layers.

        Parameters:
        -----------
        queries: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the query layer.
        keys: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the keys layer.
        values: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the values layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The attended values.

        NOTE:
        1. Follow the blueprint from class on computing the various phases of attention (i.e. A_1, A_2, A_3, and A_4).
        2. Refer to the notebook for a refresher on the big-picture equations.
        3. You will need to rely on batch matrix multiplication. The code does not differ from regular multiplication,
        but it affects the setup of the shapes.
        4. It is important to keep track of shapes at the various phases. I suggest keeping track of shapes in code
        comments above each line of code you write.
        5. Don't forget that you pre-computed the attention gain.
        6. Don't forget to incorporate the causal mask to implement causal attention (if that option is turned on).
        The function `tril` from tf_util should be very helpful.11
        '''
        B, T, H = tf.shape(queries)[0], tf.shape(queries)[1], self.units
        A       = self.num_heads
        head_dim = H // A

        # 1) reshape & transpose to (B, A, T, head_dim)
        def split_heads(x):
            x = tf.reshape(x, [B, T, A, head_dim])
            return tf.transpose(x, [0, 2, 1, 3])

        Q = split_heads(queries)   # (B, A, T, D)
        K = split_heads(keys)      # (B, A, T, D)
        V = split_heads(values)    # (B, A, T, D)

        # 2) compute raw attention scores A1 = Q·Kᵀ  shape (B, A, T, T), * gain
        #    note: batched matmul handles the 4-D case automatically
        A1 = tf.matmul(Q, K, transpose_b=True) * self.gain

        # 3) apply causal mask (if enabled)
        if self.causal:
            # tril gives a (T, T) mask of 1s on & below diag
            mask = tril(T)                          # (T,T)
            mask = tf.reshape(mask, [1, 1, T, T])   # (1,1,T,T)
            # wherever mask == 0, set score to -inf
            A1 = tf.where(mask==0,
                          tf.fill(tf.shape(A1), tf.constant(-1e9, dtype=A1.dtype)),
                          A1)

        # 4) softmax → (B, A, T, T)
        A2 = tf.nn.softmax(A1, axis=-1)

        # 5) dropout → A3
        A3 = self.attn_dropout(A2)

        # 6) attention output A4 = A3 · V → shape (B, A, T, D)
        A4 = tf.matmul(A3, V)

        # 7) recombine heads: back to (B, T, H_qkv)
        def combine_heads(x):
            x = tf.transpose(x, [0, 2, 1, 3])        # (B, T, A, D)
            return tf.reshape(x, [B, T, H])

        return combine_heads(A4)


class MultiHeadAttentionBlock(block.Block):
    '''Block that encapsulates MultiHeadAttention and related blocks. Here is a summary of the block:

    QueryKeyValueBlock → MultiHead Attention → Dense → Dropout

    All the layers/subblocks have H (i.e. num_embed) neurons. The Dense layer uses He init and a linear act fun.

    NOTE: The Dense layer in this block (according to the paper) does NOT use layer norm.
    '''
    def __init__(self, blockname, num_heads, units, prev_layer_or_block, dropout_rate=0.1, causal=True):
        '''MultiHeadAttentionBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        num_heads: int.
            Number of attention heads to use in the attention block.
        units: int.
            Number of neurons in the attention block (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer that is in the attention block that is applied to the
            attention values. The dropout rate is the same for the dropout layer in this block and the attention
            subblock.
        causal: bool.
            Whether to apply a causal mask to remove/mask out the ability for the layer to pay attention to tokens
            that are in the future of the current one in the sequence.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        pass


    def __call__(self, x):
        '''Forward pass through the MultiHead Attention Block.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        pass


class MLPBlock(block.Block):
    '''MLP block that tends to follow the attention block. Composed of the following layers:

    Dense → Dense → Dropout

    Implements a bottleneck design: 1st Dense layer has 4x the units and the 2nd Dense layer has 1x.

    1st Dense layer:
    ----------------
    - Uses the gelu activation function, layernorm

    2nd Dense layer:
    ----------------
    - Uses the linear/identity activation function, no layernorm
    '''
    def __init__(self, blockname, units, prev_layer_or_block, exp_factor=4, dropout_rate=0.1):
        '''MLPBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units: int.
            Number of neurons in the MLP block dense layers (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        exp_factor: int.
            The expansion factor that scales the number of units in the 1st Dense layer. Controls how large the
            bottleneck is in the block.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        pass

    def __call__(self, x):
        '''Forward pass through the MLPBlock with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        pass


class TransformerBlock(block.Block):
    '''The Transformer Block, composed of a single MultiHeadAtention Block followed by a single MLP Block.'''
    def __init__(self, blockname, units, num_heads, prev_layer_or_block, dropout_rate=0.1):
        '''TransformerBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units: int.
            Number of neurons in the Transformer block (H — i.e. embed_dim).
        num_heads: int.
            Number of attention heads to use in the attention block.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use throughout the block.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        pass

    def __call__(self, x):
        '''Forward pass through the Transformer block with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs

        NOTE: Don't forget the residual connections that allows the input to skip to the end of each block.
        '''
        pass


class PositionalEncodingBlock(block.Block):
    '''Block that combines PositionalEncoding layer and a Dropout layer in the following order:

    PositionalEncoding → Dropout
    '''
    def __init__(self, blockname, embed_dim, prev_layer_or_block, dropout_rate=0.1):
        '''PositionalEncodingBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        embed_dim: int.
            Number of neurons in the Embedding layer (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the dropout layer1.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers.
        '''
        pass

    def __call__(self, x):
        '''Forward pass through the block with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        pass
