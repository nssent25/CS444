'''bottleneck_block.py
Bottleneck block used to make deeper ResNets (ResNet-50+)
Nithun Selva and Saad Khan
CS444: Deep Learning
Project 2: Branch Neural Networks
'''
import tensorflow as tf

import block
from layers import Conv2D
from inception_layers import Conv2D1x1


class BottleneckBlock(block.Block):
    '''The Bottleneck Block used in ResNet-50 and deeper variants. Contains two parallel branches:

    Main branch: Three conv layers:
    1) 1x1 Conv2D to reduce dimensions (bottleneck)
    2) 3x3 Conv2D at reduced dimensions 
    3) 1x1 Conv2D to expand dimensions back

    Residual branch: The input to the block `x` (with optional 1x1 conv if needed)

    The outputs from both branches are summed and ReLU is applied.
    '''
    def __init__(self, blockname, units, prev_layer_or_block, expansion=4, strides=1):
        '''BottleneckBlock constructor.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block. Used for debugging/printing summary of net.
        units: int.
            Number of output units/filters in the final 1x1 conv of the main branch.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object.
        expansion: int.
            Expansion factor for the bottleneck. 
        strides: int.
            The convolutional stride for the 3x3 conv in the main branch.
        '''
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)
        self.strides = strides
        self.expansion = expansion
        # print(f"Creating {blockname} with expansion {expansion} and strides {strides}")
        self.bottleneck_units = units // expansion

        # Main branch
        # 1. 1x1 conv to reduce dimensions
        self.conv_reduce = Conv2D1x1(
            name=f"{blockname}/conv_reduce",
            units=self.bottleneck_units,
            activation='relu',
            prev_layer_or_block=prev_layer_or_block,
            do_batch_norm=True
        )
        
        # 2. 3x3 conv at reduced dimensions
        self.conv_3x3 = Conv2D(
            name=f"{blockname}/conv_3x3",
            units=self.bottleneck_units,
            kernel_size=(3, 3),
            activation='relu',
            prev_layer_or_block=self.conv_reduce,
            wt_init='he',
            strides=strides,
            do_batch_norm=True
        )
        
        # 3. 1x1 conv to expand dimensions back
        self.conv_expand = Conv2D1x1(
            name=f"{blockname}/conv_expand",
            units=units,
            activation='linear', 
            prev_layer_or_block=self.conv_3x3,
            do_batch_norm=True
        )
        
        self.layers = [self.conv_reduce, self.conv_3x3, self.conv_expand]
        
        # Skip connection (1x1 conv if needed) when:
        # stride > 1 (spatial dimensions change) or
        # input channels != output channels
        self.need_projection = strides > 1 or self._get_input_units(prev_layer_or_block) != units
        
        if self.need_projection:
            self.skip_branch = Conv2D1x1(
                name=f"{blockname}/skip_conv1x1",
                units=units,
                activation='linear',
                prev_layer_or_block=prev_layer_or_block,
                do_batch_norm=True,
                strides=strides
            )
            self.layers.append(self.skip_branch)

    def _get_input_units(self, prev_layer_or_block):
        """Helper method to get the number of units/channels from the previous layer"""
        if prev_layer_or_block is None:
            return None
        
        if hasattr(prev_layer_or_block, 'units'):
            return prev_layer_or_block.units
        else:
            return None  # Will need projection shortcut if we can't determine

    def __call__(self, x):
        '''Forward pass through the Bottleneck Block.'''
        # Main branch
        net_act = self.conv_reduce(x)
        net_act = self.conv_3x3(net_act)
        net_act = self.conv_expand(net_act)
        
        # Skip connection
        if self.need_projection:
            skip = self.skip_branch(x)
        else:
            skip = x
            
        # Sum and apply ReLU activation
        net_act = tf.nn.relu(net_act + skip)
        return net_act

    def __str__(self):
        '''Custom Bottleneck Block toString method.'''
        string = self.blockname + ':'
        for layer in reversed(self.layers[:-1]):  # All main branch layers
            string += '\n\t' + layer.__str__()
            
        if self.need_projection:
            string += '\n\t-->' + self.layers[-1].__str__() + '-->'
            
        return string