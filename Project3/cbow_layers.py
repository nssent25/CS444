'''cbow_layers.py
New neural network layers used for the CBOW Network
Nithun Selva and Saad Khan
CS444: Deep Learning
Project 3: Word Embeddings
'''
import tensorflow as tf

import layers

class DenseEmbedding(layers.Dense):
    '''A DenseEmbedding layer, which is just like a regular Dense layer, except it interprets each mini-batch as
    a collection of INDICES to select or pull out rows of the Dense weight matrix. These extracted rows extracted from
    the weight matrix are net_in/net_act (after adding the bias).
    '''
    def __init__(self, name, units, prev_layer_or_block=None):
        '''DenseEmbedding constructor

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        units. int.
            The number of neurons in the layer H.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        You should only need to call and pass in relevant information into the superclass constructor to implement this
        method.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input for the current DenseEmbedding layer.

        Parameters:
        -----------
        x: tf.constant. tf.int32s. shape=(B,).
            Mini-batch of indices of weights to extract for the net input computation.
            NOTE: On 1st compile forward pass, the shape is actually (1, M) to make sure the weights get initialized
            correctly.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            The net_in.

        NOTE: Don't forget the bias!
        '''
        # KEEP THE FOLLOWING.
        # NOTE: You need to write some code after the if statement.
        if self.wts is None:
            self.init_params(input_shape=x.shape)
            # Special case to only handle lazy wt/bias initialization during net compile
            return x @ self.wts + self.b

