'''transformer_layers.py
Layers related to transformer neural networks.
Saad Khan and Nithun Selva
CS 444: Deep Learning
'''
import tensorflow as tf

import layers
from tf_util import interleave_cols

class Embedding(layers.Layer):
    '''Embedding layer. Takes a mini-batch of ints and for net_in extracts the weights at the specified indices.'''
    def __init__(self, name, input_dim, embed_dim, prev_layer_or_block=None):
        '''Embedding layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        input_dim. int.
            The number of neurons in the input layer `M`.
        embed_dim. int.
            The number of neurons in the current layer `H`.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Initialize the layer's parameters.
        '''
        
        # Call the superclass constructor
        super().__init__(name, activation='linear', prev_layer_or_block=prev_layer_or_block)
        # Initialize the layer's parameters
        self.init_params(input_dim, embed_dim)
        
    def has_wts(self):
        '''Returns whether the Embedding layer has weights. It does...'''
        return True

    def init_params(self, input_dim, embed_dim):
        '''Initializes the Embedding layer's weights. There should be no bias.

        Parameters:
        -----------
        input_dim: int.
            Number of neurons in the Input layer (`M`).
        embed_dim: int.
            Number of neurons in the current layer (`H`).

        NOTE:
        - Remember to turn off the bias.
        - Use He initialization.
        '''

        std_dev = tf.sqrt(self.get_kaiming_gain() / tf.cast(input_dim, tf.float32))
        self.b = None
        self.wts = tf.Variable(tf.random.normal([input_dim, embed_dim], mean=0.0, stddev=std_dev))
        
    def compute_net_input(self, x):
        '''Computes the net input for the current Embedding layer.

        Parameters:
        -----------
        x: tf.constant. tf.int32. shape=(B, T).
            Mini-batch of int indices.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The net_in, which is the weights extracted at the specified indices.

        NOTE:
        - This layer does NOT use lazy initialization.
        - The presence of the time dimension should not affect your code compared to if it were not there.
        '''
        return tf.gather(self.wts, x)
    
    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Embedding layer output({self.layer_name}) shape: {self.output_shape}'


class PositionalEncoding(layers.Layer):
    '''Positional Encoding layer that implements sin/cos position coding.'''
    def __init__(self, name, embed_dim, prev_layer_or_block=None):
        '''PositionalEncoding layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        embed_dim. int.
            The number of neurons in the current layer `H` and in the Embedding layer below.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Print a warning/error if the embedding dimension (H) is not even, since this layer's sin/cos coding requires
        an even split.
        '''
        super().__init__(name, activation='linear', prev_layer_or_block=prev_layer_or_block)
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even for PositionalEncoding, but got {embed_dim}")
        self.embed_dim = embed_dim
        self.pos_encoding = None  # For lazy initialization

    def create_position_encoding(self, embed_dim, seq_len):
        '''Creates a positional encoding tensor using the sin/cos scheme for a sequence of length `seq_len` tokens
        for each of the `embed_dim`/H neurons. See notebook for a refresher on the equation.

        Parameters:
        -----------
        embed_dim: int.
            The number of neurons in the Embedding layer (H).
        seq_len: int.
            The length of sequences processed by the transformer.

        Returns:
        --------
        tf.constant. shape=(1, T, H).
            A positional encoding tensor, where the first axis is a singleton dimension to handle the batch dimension,
            T is the sequence length, and H is the number of embedding layer neurons.

        NOTE:
        - It might be helpful to think of creating sin frequences for H/2 neurons in the embedding layer and then
        creating cos frequences for the remaining H/2 neurons separately. Then after having both sets, interleaving
        them.
        - The provided `interleave_cols` function should be helpful, as should be tf.expand_dims.
        - To allow TensorFlow track the flow of gradients, you should implement this with 100% TensorFlow and no loops.
        '''
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]  # Shape: (T, 1)

        # k_values are the actual k in the formula: 0, 0, 2, 2, ..., H-2, H-2
        # We need k for each of the H/2 unique frequencies. These are 0, 2, 4, ..., H-2
        k_for_freq = tf.range(0, embed_dim, 2, dtype=tf.float32) # Shape: (H/2)

        # omega_k = 1 / (10000^(k / H_embed))
        # div_term corresponds to omega_k for each of the H/2 frequencies
        div_term = 1.0 / tf.pow(10000.0, k_for_freq / tf.cast(embed_dim, tf.float32)) # Shape: (H/2)
        div_term = div_term[tf.newaxis, :] # Shape: (1, H/2) for broadcasting

        # angle_rads = t * omega_k
        angle_rads = positions * div_term  # Shape: (T, H/2)
        sin_component = tf.sin(angle_rads)  # Shape: (T, H/2)
        cos_component = tf.cos(angle_rads)  # Shape: (T, H/2)

        # Interleave: sin at even indices (0, 2, ...), cos at odd indices (1, 3, ...)
        pe = interleave_cols(sin_component, cos_component)  # Shape: (T, H)
        
        pe = tf.expand_dims(pe, axis=0)  # Shape: (1, T, H) to allow broadcasting with batch
        return pe

    def compute_net_input(self, x):
        '''Computes the net input for the current PositionalEncoding layer, which is the sum of the input with the
        position coding tensor.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            Input from the layer beneath in the network.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The net_in, the input with position coding added.

        NOTE: This layer uses lazy initialization. This means that if the position code has not been defined yet,
        we call `create_position_encoding` to create it and set the result to the instance variable.
        '''
        B, T, H = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Lazy init
        if self.pos_encoding is None:
            self.pos_encoding = self.create_position_encoding(seq_len=T, embed_dim=self.embed_dim)
        
        return x + self.pos_encoding

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Positional encoding layer output({self.layer_name}) shape: {self.output_shape}'
