'''vgg_nets.py
The family of VGG neural networks implemented using the CS444 deep learning library
Saad Khan and Nithun Selva
CS444: Deep Learning
'''
import network
from layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from block import VGGConvBlock, VGGDenseBlock


class VGG4(network.DeepNetwork):
    '''The VGG4 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D → Flatten → Dense → Dropout → Dense

    Notes:
    1. All convolutions are 3x3 in the VGG family of neural networks.
    2. All max pooling windows are 2x2 in the VGG family of neural networks.
    3. The dropout rate is `0.5`.
    4. The activation used in `Conv2D` and hidden `Dense` layers in the VGG family of neural networks are ReLU.
    5. The output layer should use softmax activation.
    '''
    def __init__(self, C, input_feats_shape, filters=64, dense_units=128, reg=0, wt_scale=1e-3, wt_init='normal'):
        '''VGG4 network constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in each convolutional layer (the same in all layers).
        dense_units: int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore Week 1 and until instructed otherwise.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common with VGG4.
        2. Build out the VGG4 network using the layers that you made.
        3. Don't forget to link up your layers with the `prev_layer_or_block` keyword. For the 1st layer in the net,
        you set this to `None`.

        NOTE:
        - To make sure you configure everything correctly, make it a point to override every keyword argment in each of
        the layers with appropriate values (even if they are no different than the defaults). Once you get more familar
        with the deep learning library, you may decide to skip setting some defaults in the coming weeks.
        - The only requirement on your variable names is that you MUST name your output layer `self.output_layer`.
        - Use helpful names for your layers and variables. You will have to live with them!
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        
        # Convolutional layers
        self.conv1 = Conv2D(name="Conv2D_1", units=filters, kernel_size=(3, 3), prev_layer_or_block=None, activation='relu', wt_scale=wt_scale, wt_init=wt_init)
        self.conv2 = Conv2D(name="Conv2D_2", units=filters, kernel_size=(3, 3), prev_layer_or_block=self.conv1, activation='relu', wt_scale=wt_scale, wt_init=wt_init)
        
        # Max pooling layer
        self.pool1 = MaxPool2D(name="MaxPool_1", pool_size=(2, 2), strides=2, prev_layer_or_block=self.conv2)

        # Flatten layer
        self.flatten = Flatten(name="Flatten", prev_layer_or_block=self.pool1)

        # Fully connected layer
        self.dense1 = Dense(name="Dense_1", units=dense_units, prev_layer_or_block=self.flatten, activation='relu', wt_scale=wt_scale, wt_init=wt_init)

        # Dropout layer
        self.dropout = Dropout(name="Dropout", rate=0.5, prev_layer_or_block=self.dense1)

        # Output layer
        self.output_layer = Dense(name="Output", units=C, prev_layer_or_block=self.dropout, activation='softmax', wt_scale=wt_scale, wt_init=wt_init)

    def __call__(self, x):
        '''Forward pass through the VGG4 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        NOTE: Use the functional API to perform the forward pass through your network!
        '''
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        return x 


class VGG6(network.DeepNetwork):
    '''The VGG6 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D → Conv2D → Conv2D → MaxPool2D → Flatten → Dense → Dropout → Dense

    Aside from differences in the number of units in various layers, all hard-coded hyperparameters from VGG4 carry over
    to VGG6 (e.g. 3x3 convolutions, ReLU activations in conv layers, etc.).
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128), dense_units=(256,), reg=0, wt_scale=1e-3,
                 wt_init='normal'):
        '''The VGG6 constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until instructed otherwise.

        TODO: Use blocks to build the VGG6 network (where appropriate). For grading purposes, do NOT use ONLY Layer
        objects here! The number of lines of code should be comparable or perhaps a little less than VGG4
        (thanks to blocks!).
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
    
        # First conv block
        self.conv_block1 = VGGConvBlock(
            blockname="ConvBlock1",
            units=filters[0],
            prev_layer_or_block=None,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init
        )

        # Second conv block
        self.conv_block2 = VGGConvBlock(
            blockname="ConvBlock2",
            units=filters[1],
            prev_layer_or_block=self.conv_block1,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init
        )

        # Flatten layer
        self.flatten = Flatten(name="Flatten",prev_layer_or_block=self.conv_block2)

        # Dense block with dropout
        self.dense_block = VGGDenseBlock(
            blockname="DenseBlock1",
            units=dense_units,
            prev_layer_or_block=self.flatten,
            num_dense_blocks=1,
            wt_scale=wt_scale,
            dropout=True,
            dropout_rate=0.5,
            wt_init=wt_init
        )

        # Output layer
        self.output_layer = Dense(name="Output", units=C, prev_layer_or_block=self.dense_block, activation='softmax', wt_scale=wt_scale, wt_init=wt_init)

    def __call__(self, x):
        '''Forward pass through the VGG6 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        NOTE: Use the functional API to perform the forward pass through your network!
        '''
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        x = self.output_layer(x)
    
        return x


class VGG8(network.DeepNetwork):
    '''The VGG8 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D → Conv2D → Conv2D → MaxPool2D → Conv2D → Conv2D → MaxPool2D → Flatten → Dense → Dropout → Dense

    Aside from differences in the number of units in various layers, all hard-coded hyperparameters from VGG4 carry over
    to VGG6 (e.g. 3x3 convolutions, ReLU activations in conv layers, etc.).
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256), dense_units=(512,), reg=0, wt_scale=1e-3,
                 wt_init='he', conv_dropout=False, conv_dropout_rates=(0.1, 0.2, 0.3)):
        '''The VGG8 constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer (if using normal wt init method).
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        conv_dropout: bool.
            Do we place a dropout layer in each conv block?
        conv_dropout_rates: tuple of floats. len(conv_dropout_rates)=num_conv_blocks
            The dropout rate to use in each conv block. Only has an effect if `conv_dropout` is True.

        TODO: Use blocks to build the VGG8 network (where appropriate). For grading purposes and your sanity, do NOT use
        ONLY Layer objects here!
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
    
        # First conv block
        self.conv_block1 = VGGConvBlock(
            blockname="ConvBlock1",
            units=filters[0],
            prev_layer_or_block=None,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[0] if conv_dropout else 0
        )

        # Second conv block
        self.conv_block2 = VGGConvBlock(
            blockname="ConvBlock2",
            units=filters[1],
            prev_layer_or_block=self.conv_block1,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[1] if conv_dropout else 0
        )

        # Third conv block
        self.conv_block3 = VGGConvBlock(
            blockname="ConvBlock3",
            units=filters[2],
            prev_layer_or_block=self.conv_block2,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[2] if conv_dropout else 0
        )

        # Flatten layer
        self.flatten = Flatten(name="Flatten",prev_layer_or_block=self.conv_block3)

        # Dense block with dropout
        self.dense_block = VGGDenseBlock(
            blockname="DenseBlock1",
            units=dense_units,
            prev_layer_or_block=self.flatten,
            num_dense_blocks=1,
            wt_scale=wt_scale,
            dropout=True,
            dropout_rate=0.5,
            wt_init=wt_init
        )

        # Output layer
        self.output_layer = Dense(name="Output", units=C, prev_layer_or_block=self.dense_block, activation='softmax', wt_scale=wt_scale, wt_init=wt_init)

    def __call__(self, x):
        '''Forward pass through the VGG8 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        NOTE: Use the functional API to perform the forward pass through your network!
        '''
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        x = self.output_layer(x)
    
        return x


class VGG15(network.DeepNetwork):
    '''The VGG15 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Flatten →
    Dense → Dropout →
    Dense

    Aside from differences in the number of units in various layers, all hard-coded hyperparameters from VGG4 carry over
    to VGG6 (e.g. 3x3 convolutions, ReLU activations in conv layers, etc.).
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256, 512, 512), dense_units=(512,), reg=0.6,
                 wt_scale=1e-3, wt_init='he', conv_dropout=False, conv_dropout_rates=(0.1, 0.2, 0.3, 0.3, 0.3)):
        '''The VGG15 constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer (if using normal wt init method).
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        conv_dropout: bool.
            Do we place a dropout layer in each conv block?
        conv_dropout_rates: tuple of floats. len(conv_dropout_rates)=num_conv_blocks
            The dropout rate to use in each conv block. Only has an effect if `conv_dropout` is True.

        TODO: Use blocks to build the VGG15 network (where appropriate). For grading purposes and your sanity, do NOT
        use ONLY Layer objects here!
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        
        # First conv block
        self.conv_block1 = VGGConvBlock(
            blockname="ConvBlock1",
            units=filters[0],
            prev_layer_or_block=None,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init
        )

        # Second conv block
        self.conv_block2 = VGGConvBlock(
            blockname="ConvBlock2",
            units=filters[1],
            prev_layer_or_block=self.conv_block1,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init
        )

        # Third conv block
        self.conv_block3 = VGGConvBlock(
            blockname="ConvBlock3",
            units=filters[2],
            prev_layer_or_block=self.conv_block2,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init
        )

        # Fourth conv block
        self.conv_block4 = VGGConvBlock(
            blockname="ConvBlock4",
            units=filters[3],
            prev_layer_or_block=self.conv_block3,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init
        )

        # Fifth conv block
        self.conv_block5 = VGGConvBlock(
            blockname="ConvBlock5",
            units=filters[4],
            prev_layer_or_block=self.conv_block4,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init
        )

        # Flatten layer
        self.flatten = Flatten(name="Flatten",prev_layer_or_block=self.conv_block5)

        # Dense block with dropout
        self.dense_block = VGGDenseBlock(
            blockname="DenseBlock1",
            units=dense_units,
            prev_layer_or_block=self.flatten,
            num_dense_blocks=1,
            wt_scale=wt_scale,
            dropout=True,
            dropout_rate=0.5,
            wt_init=wt_init
        )

        # Output layer
        self.output_layer = Dense(name="Output", units=C, prev_layer_or_block=self.dense_block, activation='softmax', wt_scale=wt_scale, wt_init=wt_init)
    
    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        NOTE: Use the functional API to perform the forward pass through your network!
        '''
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        x = self.output_layer(x)
    
        return x


class VGG4Plus(network.DeepNetwork):
    '''The VGG4 network with batch normalization added to all Conv2D layers and all non-output Dense layers.'''
    def __init__(self, C, input_feats_shape, filters=64, dense_units=128, reg=0, wt_scale=1e-3, wt_init='he'):
        '''VGG4Plus network constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in each convolutional layer (the same in all layers).
        dense_units: int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        
        # Convolutional layers with batch normalization
        self.conv1 = Conv2D(
            name="Conv2D_1", 
            units=filters, 
            kernel_size=(3, 3), 
            prev_layer_or_block=None, 
            activation='relu', 
            wt_scale=wt_scale, 
            wt_init=wt_init,
            do_batch_norm=True
        )
        
        self.conv2 = Conv2D(
            name="Conv2D_2", 
            units=filters, 
            kernel_size=(3, 3), 
            prev_layer_or_block=self.conv1, 
            activation='relu', 
            wt_scale=wt_scale, 
            wt_init=wt_init,
            do_batch_norm=True
        )
        
        # Max pooling layer
        self.pool1 = MaxPool2D(
            name="MaxPool_1", 
            pool_size=(2, 2), 
            strides=2, 
            prev_layer_or_block=self.conv2
        )

        # Flatten layer
        self.flatten = Flatten(
            name="Flatten", 
            prev_layer_or_block=self.pool1
        )

        # Fully connected layer with batch normalization
        self.dense1 = Dense(
            name="Dense_1", 
            units=dense_units, 
            prev_layer_or_block=self.flatten, 
            activation='relu', 
            wt_scale=wt_scale, 
            wt_init=wt_init,
            do_batch_norm=True
        )

        # Dropout layer
        self.dropout = Dropout(
            name="Dropout", 
            rate=0.5, 
            prev_layer_or_block=self.dense1
        )

        # Output layer (no batch norm in output layer)
        self.output_layer = Dense(
            name="Output", 
            units=C, 
            prev_layer_or_block=self.dropout, 
            activation='softmax', 
            wt_scale=wt_scale, 
            wt_init=wt_init,
            do_batch_norm=False
        )

    def __call__(self, x):
        '''Forward pass through the VGG4 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        NOTE: Use the functional API to perform the forward pass through your network!
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        return x


class VGG15Plus(network.DeepNetwork):
    '''The VGG15Plus network is the VGG15 network with batch normalization added to all Conv2D layers and all
    non-output Dense layers.
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256, 512, 512), dense_units=(512,), reg=0.6,
                  wt_scale=1e-3, wt_init='he', conv_dropout=False, conv_dropout_rates=(0.1, 0.2, 0.3, 0.3, 0.3)):
        '''The VGG15Plus constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer (if using normal wt init method).
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        conv_dropout: bool.
            Do we place a dropout layer in each conv block?
        conv_dropout_rates: tuple of floats. len(conv_dropout_rates)=num_conv_blocks
            The dropout rate to use in each conv block. Only has an effect if `conv_dropout` is True.
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        
        # First conv block with batch normalization
        self.conv_block1 = VGGConvBlock(
            blockname="ConvBlock1",
            units=filters[0],
            prev_layer_or_block=None,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[0]
        )

        # Second conv block with batch normalization
        self.conv_block2 = VGGConvBlock(
            blockname="ConvBlock2",
            units=filters[1],
            prev_layer_or_block=self.conv_block1,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[1]
        )

        # Third conv block with batch normalization
        self.conv_block3 = VGGConvBlock(
            blockname="ConvBlock3",
            units=filters[2],
            prev_layer_or_block=self.conv_block2,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[2]
        )        
                
        # Fourth conv block with batch normalization
        self.conv_block4 = VGGConvBlock(
            blockname="ConvBlock4",
            units=filters[3],
            prev_layer_or_block=self.conv_block3,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[3]
        )

        # Fifth conv block with batch normalization
        self.conv_block5 = VGGConvBlock(
            blockname="ConvBlock5",
            units=filters[4],
            prev_layer_or_block=self.conv_block4,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[4]
        )

        # Flatten layer
        self.flatten = Flatten(
            name="Flatten",
            prev_layer_or_block=self.conv_block5
        )

        # Dense block with batch normalization and dropout
        self.dense_block = VGGDenseBlock(
            blockname="DenseBlock1",
            units=dense_units,
            prev_layer_or_block=self.flatten,
            num_dense_blocks=1,
            wt_scale=wt_scale,
            dropout=True,
            dropout_rate=0.5,
            wt_init=wt_init,
            do_batch_norm=True
        )

        # Output layer (no batch norm)
        self.output_layer = Dense(
            name="Output", 
            units=C, 
            prev_layer_or_block=self.dense_block, 
            activation='softmax', 
            wt_scale=wt_scale, 
            wt_init=wt_init,
            do_batch_norm=False
        )

    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        NOTE: Use the functional API to perform the forward pass through your network!
        '''
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        x = self.output_layer(x)
    
        return x


class VGG15PlusPlus(network.DeepNetwork):
    '''The VGG15PlusPlus network is the VGG15 network with:
    1. Batch normalization added to all Conv2D layers and all non-output Dense layers.
    2. Dropout added to all conv blocks.
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256, 512, 512), dense_units=(512,), reg=0.6,
                 wt_scale=1e-3, wt_init='he', conv_dropout=True, conv_dropout_rates=(0.3, 0.4, 0.4, 0.4, 0.4)):
        
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)

        # First conv block with batch normalization and dropout
        self.conv_block1 = VGGConvBlock(
            blockname="ConvBlock1",
            units=filters[0],
            prev_layer_or_block=None,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[0]
        )

        # Second conv block with batch normalization and dropout
        self.conv_block2 = VGGConvBlock(
            blockname="ConvBlock2",
            units=filters[1],
            prev_layer_or_block=self.conv_block1,
            num_conv_layers=2,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[1]
        )

        # Third conv block with batch normalization and dropout
        self.conv_block3 = VGGConvBlock(
            blockname="ConvBlock3",
            units=filters[2],
            prev_layer_or_block=self.conv_block2,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[2]
        )

        # Fourth conv block with batch normalization and dropout
        self.conv_block4 = VGGConvBlock(
            blockname="ConvBlock4",
            units=filters[3],
            prev_layer_or_block=self.conv_block3,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[3]
        )

        # Fifth conv block with batch normalization and dropout
        self.conv_block5 = VGGConvBlock(
            blockname="ConvBlock5",
            units=filters[4],
            prev_layer_or_block=self.conv_block4,
            num_conv_layers=3,
            wt_scale=wt_scale,
            wt_init=wt_init,
            do_batch_norm=True,
            dropout=conv_dropout,
            dropout_rate=conv_dropout_rates[4]
        )

        # Flatten layer
        self.flatten = Flatten(
            name="Flatten",
            prev_layer_or_block=self.conv_block5
        )

        # Dense block with batch normalization and dropout
        self.dense_block = VGGDenseBlock(
            blockname="DenseBlock1",
            units=dense_units,
            prev_layer_or_block=self.flatten,
            num_dense_blocks=1,
            wt_scale=wt_scale,
            dropout=True,
            dropout_rate=0.5,
            wt_init=wt_init,
            do_batch_norm=True
        )

        # Output layer (no batch norm)
        self.output_layer = Dense(
            name="Output", 
            units=C, 
            prev_layer_or_block=self.dense_block, 
            activation='softmax', 
            wt_scale=wt_scale, 
            wt_init=wt_init,
            do_batch_norm=False
        )

    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        NOTE: Use the functional API to perform the forward pass through your network!
        '''
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        x = self.output_layer(x)
    
        return x
