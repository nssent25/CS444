'''resnets.py
Various neural networks in the ResNet family
Nithun Selva and Saad Khan
CS 444: Deep Learning
Project 2: Branch Neural Networks
'''
import tensorflow as tf

import network
from layers import Conv2D, Dense
from inception_layers import GlobalAveragePooling2D
from residual_block import ResidualBlock
from bottleneck_block import BottleneckBlock

from resnext_block import ResNeXtBlock



def stack_residualblocks(stackname, units, num_blocks, prev_layer_or_block, first_block_stride=1, block_type='residual', expansion=4, cardinality=32):
    '''Creates a stack of `num_blocks` Residual Blocks, each with `units` neurons.

    Parameters:
    -----------
    stackname: str.
        Human-readable name for the current stack of Residual Blocks
    units: int.
        Number of units in each block in the stack.
    num_blocks: int.
        Number of blocks to create as part of the stack.
    prev_layer_or_block: Layer (or Layer-like) object.
        Reference to the Layer/Block object that is beneath the first block. `None` if there is no preceding
        layer/block.
    first_block_stride: int. 1 or 2.
        The stride on the 1st block in a stack could EITHER be 1 or 2.
        The stride for ALL blocks in the stack after the first ALWAYS is 1.
    block_type: str.
        Ignore for base project. Option here to help in case you want to build very deep ResNets (e.g. ResNet-50)
        for Extensions, which use 'bottleneck' blocks.
    expansion: int.
        Expansion factor for bottleneck blocks. Only used when block_type='bottleneck'.

    Returns:
    --------
    Python list.
        lis Residualt of Residual Blocks in the current stac
    NOTE: To help keep stacks, blocks, and layers organized when printing the summary, modify each block name by
    preprending the stack name to which it belongs. For example, if this is stack_1, call the first two blocks
    'stack_1/block_1' and 'stack_1/block_2'.
k.

    NOTE: To help keep stacks, blocks, and layers organized when printing the summary, modify each block name by
    preprending the stack name to which it belongs. For example, if this is stack_1, call the first two blocks
    'stack_1/block_1' and 'stack_1/block_2'.
    '''
    blocks = []
    for i in range(num_blocks):
        blockname = f"{stackname}/block_{i+1}"
        stride = first_block_stride if i == 0 else 1  # Apply stride only to the first block

        if block_type == 'residual':
            block = ResidualBlock(
                blockname=blockname,
                units=units,
                prev_layer_or_block=prev_layer_or_block,
                strides=stride
            )
        elif block_type == 'bottleneck':
            block = BottleneckBlock(
                blockname=blockname,
                units=units,
                prev_layer_or_block=prev_layer_or_block,
                expansion=expansion,
                strides=stride
            )
        elif block_type == 'resnext':
            block = ResNeXtBlock(
                blockname=blockname,
                units=units,
                prev_layer_or_block=prev_layer_or_block,
                cardinality=cardinality,
                expansion=expansion,
                strides=stride,
            )
        else:
            raise ValueError(f"Unknown block_type: {block_type}. Use 'residual' or 'bottleneck'.")
        blocks.append(block)
        prev_layer_or_block = block  # Update prev_layer_or_block for the next block

    return blocks





class ResNet(network.DeepNetwork):
    '''ResNet parent class
    '''
    def __call__(self, x):
        '''Forward pass through the ResNet with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        Hint: you are storing all layers/blocks sequentially in self.layers and there are NO skip connections acrossblocks ;)
        '''
        for layer in self.layers:
            x = layer(x)
        return x

    def summary(self):
        '''Custom toString method for ResNets'''
        print(75*'-')
        for layer in reversed(self.layers):
            print(layer)


class ResNet8(ResNet):
    '''The ResNet8 network. Here is an overview of its structure:

    Conv2D → 3xResidualBlocks → GlobalAveragePooling2D → Dense

    Layer/block properties:
    -----------------------
    - Conv layer: 3x3 kernel size. ReLU activation. He initialization (always!). Uses batch norm.
    - ResidualBlocks: 1st block has stride of 1, the others have stride 2.
    - Dense: He initialization (always!)

    '''
    def __init__(self, C, input_feats_shape, filters=32, block_units=(32, 64, 128), reg=0):
        '''ResNet8 constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in the 1st 2D conv layer.
        block_units: tuple of ints.
            Number of filters in each residual block.
        reg: float.
            Regularization strength.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the ResNet network. Use of self.layers for organizing layers/blocks.
        3. Remember that although Residual Blocks have parallel branches, the macro-level ResNet layers/blocks
        are arranged sequentially.

        NOTE:
        - To make sure you configure everything correctly, make it a point to check every keyword argment in each of
        the layers/blocks.
        - The only requirement on your variable names is that you MUST name your output layer `self.output_layer`.
        - Use helpful names for your layers and variables. You will have to live with them!
        '''
        super().__init__(input_feats_shape, reg)
        self.layers = []

        # Initial Conv2D layer
        self.conv = Conv2D(
            name='Conv2D_1',
            units=filters,
            kernel_size=(3, 3),
            prev_layer_or_block=None,
            activation='relu',
            wt_init='he',
            do_batch_norm=True
        )
        self.layers.append(self.conv)
        prev_layer = self.conv

        # Stack of Residual Blocks
        # The first block has stride 1, the others have stride 2
        strides = [1, 2, 2]
        for i in range(3):
            res_block = ResidualBlock(
                blockname=f'ResidualBlock{i+1}',
                units=block_units[i],
                prev_layer_or_block=prev_layer,
                strides=strides[i]
            )
            self.layers.append(res_block)
            prev_layer = res_block

        # Global Average Pooling
        self.global_pool = GlobalAveragePooling2D(
            name='GlobalAvgPool2D',
            prev_layer_or_block=prev_layer
        )
        self.layers.append(self.global_pool)
        prev_layer = self.global_pool

        # Output Dense layer
        self.output_layer = Dense(
            name='Output',
            units=C,
            activation='softmax',
            prev_layer_or_block=prev_layer,
            wt_init='he'
        )
        self.layers.append(self.output_layer)


class ResNet18(ResNet):
    '''The ResNet18 network. Here is an overview of its structure:

    Conv2D → 4 stacks of 2 ResidualBlocks → GlobalAveragePooling2D → Dense

    Layer/block properties:
    -----------------------
    - Conv layer: 3x3 kernel size. ReLU activation. He initialization (always!). Uses batch norm.
    - Stacks of Residual Blocks: 1st stack blocks in net has stride 1, the first block in the remaining 3 stacks start
    with stride 2. Two blocks per stack.
    - Dense: He initialization (always!)
    '''
    def __init__(self, C, input_feats_shape, filters=64, block_units=(64, 128, 256, 512), reg=0):
        '''ResNet18 constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in the 1st 2D conv layer.
        block_units: tuple of ints.
            Number of filters in each residual block.
        reg: float.
            Regularization strength.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the ResNet network. Use of self.layers for organizing layers/blocks.
        3. Remember that although Residual Blocks have parallel branches, the macro-level ResNet layers/blocks
        are arranged sequentially.
        '''
        super().__init__(input_feats_shape, reg)
        self.layers = []

        # Initial Conv2D layer
        self.conv = Conv2D(
            name='conv_1',
            units=filters,
            kernel_size=(3, 3),
            prev_layer_or_block=None,
            activation='relu',
            wt_init='he',
            do_batch_norm=True
        )
        self.layers.append(self.conv)
        prev_layer = self.conv

        # Stacks of Residual Blocks
        num_blocks_per_stack = 2
        first_block_strides = [1, 2, 2, 2]  # Stride for the first block in each stack

        for i in range(4):  # 4 stacks
            stack_name = f'stack_{i+1}'
            blocks = stack_residualblocks(
                stackname=stack_name,
                units=block_units[i],
                num_blocks=num_blocks_per_stack,
                prev_layer_or_block=prev_layer,
                first_block_stride=first_block_strides[i]
            )
            self.layers.extend(blocks)
            prev_layer = blocks[-1]  # Last block in the stack becomes the prev_layer for the next stack

        # Global Average Pooling
        self.global_pool = GlobalAveragePooling2D(
            name='global_avg_pool',
            prev_layer_or_block=prev_layer
        )
        self.layers.append(self.global_pool)
        prev_layer = self.global_pool

        # Output Dense layer
        self.output_layer = Dense(
            name='output',
            units=C,
            activation='softmax',
            prev_layer_or_block=prev_layer,
            wt_init='he'
        )
        self.layers.append(self.output_layer)


class ResNet50(ResNet):
    '''The ResNet50 network with bottleneck blocks. Structure:
    
    Conv2D → 4 stacks of Bottleneck Blocks → GlobalAveragePooling2D → Dense
    
    Stacks contain [3, 4, 6, 3] bottleneck blocks respectively.
    '''
    def __init__(self, C=100, input_feats_shape=(32,32,3), filters=64, block_units=(64, 128, 256, 512), reg=0):
        '''ResNet50 constructor
        
        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
        filters: int.
            Number of filters in the 1st 2D conv layer.
        block_units: tuple of ints.
            Number of output filters in the final 1x1 conv of each bottleneck block stack.
            Note: For ResNet-50, these are typically [256, 512, 1024, 2048].
        reg: float.
            Regularization strength.
        '''
        super().__init__(input_feats_shape, reg)
        self.layers = []
        
        # Initial Conv2D layer
        self.conv = Conv2D(
            name='conv_1',
            units=filters,
            kernel_size=(3, 3),
                        prev_layer_or_block=None,
            activation='relu',
            wt_init='he',
            do_batch_norm=True
        )
        self.layers.append(self.conv)
        prev_layer = self.conv
        
        # Stacks of Bottleneck Blocks
        num_blocks = [3, 4, 6, 3]  # Standard for ResNet-50
        first_block_strides = [1, 2, 2, 2]  # Stride for the first block in each stack
        
        for i in range(4):  # 4 stacks
            stack_name = f'stack_{i+1}'
            blocks = stack_residualblocks(
                stackname=stack_name,
                units=block_units[i],
                num_blocks=num_blocks[i],
                prev_layer_or_block=prev_layer,
                first_block_stride=first_block_strides[i],
                block_type='bottleneck'
            )
            self.layers.extend(blocks)
            prev_layer = blocks[-1]  # Last block becomes the prev_layer for the next stack
        
        # Global Average Pooling
        self.global_pool = GlobalAveragePooling2D(
            name='global_avg_pool',
            prev_layer_or_block=prev_layer
        )
        self.layers.append(self.global_pool)
        prev_layer = self.global_pool

        # Output Dense layer
        self.output_layer = Dense(
            name='output',
            units=C,
            activation='softmax',
            prev_layer_or_block=prev_layer,
            wt_init='he'
        )
        self.layers.append(self.output_layer)


class ResNeXt18(ResNet):
    """ResNeXt‑18: 4 stacks of 2 ResNeXtBlocks, like ResNet18 but with grouped conv."""
    def __init__(self, C, input_feats_shape=(32,32,3), reg=0,
                 cardinality=32, expansion=4):
        super().__init__(input_feats_shape, reg)
        self.layers = []

        # stem conv
        self.stem = Conv2D(
            name='conv_1',
            units=64,
            kernel_size=(3,3),
            prev_layer_or_block=None,
            activation='relu',
            wt_init='he',
            do_batch_norm=True
        )
        self.layers.append(self.stem)
        prev = self.stem

        # 4 stacks of 2 ResNeXt blocks each
        blocks_per_stack = [2,2,2,2]
        filters =      [64,128,256,512]
        first_strides =[1, 2,  2,  2]

        for i,(n_blk, f, stride) in enumerate(zip(blocks_per_stack, filters, first_strides),1):
            stk = stack_residualblocks(
                stackname=f'stack_{i}',
                units=f,
                num_blocks=n_blk,
                prev_layer_or_block=prev,
                first_block_stride=stride,
                block_type='resnext', # Use ResNeXt blocks
                cardinality=cardinality,
                expansion=expansion
            )
            self.layers.extend(stk)
            prev = stk[-1]

        # global pool + output
        self.global_pool = GlobalAveragePooling2D(
            name='global_avg_pool',
            prev_layer_or_block=prev
        )
        self.layers.append(self.global_pool)
        prev = self.global_pool

        self.output_layer = Dense(
            name='output',
            units=C,
            activation='softmax',
            prev_layer_or_block=prev,
            wt_init='he'
        )
        self.layers.append(self.output_layer)
