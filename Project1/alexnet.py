'''alexnet.py
AlexNet neural network implemented using the CS444 deep learning library
Saad Khan and Nithun Selva
CS444: Deep Learning
'''
import network
from layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense

class AlexNet(network.DeepNetwork):
    '''The AlexNet neural network architecture from "ImageNet Classification with Deep 
    Convolutional Neural Networks" (Krizhevsky et al., 2012).
    
    Original architecture:
    Conv2D(96, 11x11, stride=4) → ReLU → MaxPool(3x3, stride=2) →
    Conv2D(256, 5x5, padding=SAME) → ReLU → MaxPool(3x3, stride=2) →
    Conv2D(384, 3x3, padding=SAME) → ReLU →
    Conv2D(384, 3x3, padding=SAME) → ReLU →
    Conv2D(256, 3x3, padding=SAME) → ReLU → MaxPool(3x3, stride=2) →
    Flatten → Dense(4096) → ReLU → Dropout(0.5) →
    Dense(4096) → ReLU → Dropout(0.5) → Dense(C) → Softmax
    
    Note: This implementation uses a simplified version that works with smaller input images
    and adapts to the current framework. The original AlexNet was designed for 224x224 images.
    '''
    def __init__(self, C, input_feats_shape, reg=0, wt_scale=1e-2, wt_init='he', 
                 do_batch_norm=True):
        '''AlexNet constructor.
        
        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases when initialized.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_batch_norm: bool.
            Whether to use batch normalization.
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        
        # First block: Conv -> MaxPool
        self.conv1 = Conv2D(
            name="conv_1",
            units=96,
            kernel_size=(11, 11),
            strides=4,
            prev_layer_or_block=None,
            activation='relu',
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=do_batch_norm
        )
        
        self.pool1 = MaxPool2D(
            name="pool_1",
            pool_size=(3, 3),
            strides=2,
            prev_layer_or_block=self.conv1
        )
        
        # Second block: Conv -> MaxPool
        self.conv2 = Conv2D(
            name="conv_2",
            units=256,
            kernel_size=(5, 5),
            strides=1,
            prev_layer_or_block=self.pool1,
            activation='relu',
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=do_batch_norm
        )
        
        self.pool2 = MaxPool2D(
            name="pool_2",
            pool_size=(3, 3),
            strides=2,
            prev_layer_or_block=self.conv2
        )
        
        # Third block: 3 consecutive conv layers
        self.conv3 = Conv2D(
            name="conv_3",
            units=384,
            kernel_size=(3, 3),
            strides=1,
            prev_layer_or_block=self.pool2,
            activation='relu',
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=do_batch_norm
        )
        
        self.conv4 = Conv2D(
            name="conv_4",
            units=384,
            kernel_size=(3, 3),
            strides=1,
            prev_layer_or_block=self.conv3,
            activation='relu',
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=do_batch_norm
        )
        
        self.conv5 = Conv2D(
            name="conv_5",
            units=256,
            kernel_size=(3, 3),
            strides=1,
            prev_layer_or_block=self.conv4,
            activation='relu',
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=do_batch_norm
        )
        
        self.pool3 = MaxPool2D(
            name="pool_3",
            pool_size=(3, 3),
            strides=2,
            prev_layer_or_block=self.conv5
        )
        
        # Flatten and fully connected layers
        self.flatten = Flatten(
            name="flatten",
            prev_layer_or_block=self.pool3
        )
        
        self.dense1 = Dense(
            name="dense_1",
            units=4096,
            activation='relu',
            prev_layer_or_block=self.flatten,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=do_batch_norm
        )
        
        self.dropout1 = Dropout(
            name="dropout_1",
            rate=0.5,
            prev_layer_or_block=self.dense1
        )
        
        self.dense2 = Dense(
            name="dense_2",
            units=4096,
            activation='relu',
            prev_layer_or_block=self.dropout1,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=do_batch_norm
        )
        
        self.dropout2 = Dropout(
            name="dropout_2",
            rate=0.5,
            prev_layer_or_block=self.dense2
        )
        
        # Output layer
        self.output_layer = Dense(
            name="output",
            units=C,
            activation='softmax',
            prev_layer_or_block=self.dropout2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=False  # No batch norm in output layer
        )

    def __call__(self, x):
        '''Forward pass through the AlexNet network.
        
        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.
            
        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.
        '''
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        
        return x