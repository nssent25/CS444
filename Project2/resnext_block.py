"""
resnext_block.py
Implementation of the ResNeXt bottleneck block (grouped 3×3 conv).

Nithun Selva and Saad Khan
CS444: Deep Learning
Project 2: Branch Neural Networks
"""
import tensorflow as tf

import block
from layers import Conv2D
from inception_layers import Conv2D1x1


# --------------------------------------------------------------------------
# helper: grouped 3×3 convolution realised with ordinary Conv2D layers
# --------------------------------------------------------------------------
class _GroupedConv2D(block.Block):
    """Helper block to implement grouped 3x3 convolution using standard Conv2D layers.

    Splits the input tensor along the channel axis into `cardinality` groups,
    applies a separate 3x3 convolution to each group, and concatenates the results.
    """
    def __init__(
        self,
        blk_name,
        out_units,
        cardinality,
        strides,
        prev_layer_or_block,
    ):
        """_GroupedConv2D constructor.

        Parameters:
        -----------
        blk_name: str.
            Human-readable name for the current block.
        out_units: int.
            Total number of output units/filters for the grouped convolution.
        cardinality: int.
            Number of groups to split the input into.
        strides: int.
            The convolutional stride for the 3x3 convs.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object.
        """
        super().__init__(blk_name, prev_layer_or_block=prev_layer_or_block)

        if out_units % cardinality != 0:
            raise ValueError(
                f"out_units ({out_units}) must be divisible by cardinality "
                f"({cardinality})."
            )
        self.cardinality = cardinality
        self.group_units = out_units // cardinality

        # build one 3×3 Conv2D per group
        self.group_convs = [
            Conv2D(
                name=f"{blk_name}/grp{g+1}",
                units=self.group_units,
                kernel_size=(3, 3),
                strides=strides,
                activation="relu",
                prev_layer_or_block=prev_layer_or_block,
                wt_init="he",
                do_batch_norm=True,
            )
            for g in range(cardinality)
        ]
        self.layers = self.group_convs  # let summary() print the parts

    def __call__(self, x):
        """Forward pass for grouped convolution."""
        # slice input along channel axis → run through each conv → concat
        split = tf.split(x, self.cardinality, axis=-1)
        y = [conv(t) for conv, t in zip(self.group_convs, split)]
        return tf.concat(y, axis=-1)

    def __str__(self):
        """Custom _GroupedConv2D toString method."""
        s = self.blockname + ":"
        for layer in reversed(self.layers):
            s += "\n\t" + str(layer)
        return s


# --------------------------------------------------------------------------
# main ResNeXt bottleneck block
# --------------------------------------------------------------------------
class ResNeXtBlock(block.Block):
    """
    The ResNeXt Bottleneck Block. Similar structure to the ResNet Bottleneck Block,
    but replaces the standard 3x3 convolution with a grouped convolution.

    Layout:
        Main branch:
        1) 1x1 Conv2D to reduce dimensions (bottleneck)
        2) Grouped 3x3 Conv2D at reduced dimensions
        3) 1x1 Conv2D to expand dimensions back

        Residual branch: The input to the block `x` (with optional 1x1 conv if needed)

        The outputs from both branches are summed and ReLU is applied.
    """

    def __init__(
        self,
        blockname,
        units,
        prev_layer_or_block,
        cardinality=32,
        expansion=4,
        strides=1,
    ):
        """ResNeXtBlock constructor.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block.
        units: int.
            Number of output units/filters in the final 1x1 conv of the main branch.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object.
        cardinality: int.
            Number of groups for the grouped convolution.
        expansion: int.
            Expansion factor for the bottleneck.
        strides: int.
            The convolutional stride for the grouped 3x3 conv in the main branch.
        """
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)

        self.cardinality = cardinality
        self.expansion = expansion
        self.strides = strides

        # Main branch
        # 1. 1x1 conv to reduce dimensions
        self.bottleneck_units = units // expansion
        if self.bottleneck_units % cardinality != 0:
            raise ValueError(
                f"units/expansion ({self.bottleneck_units}) must be divisible "
                f"by cardinality ({cardinality})."
            )

        self.conv_reduce = Conv2D1x1(
            name=f"{blockname}/conv_reduce",
            units=self.bottleneck_units,
            activation="relu",
            prev_layer_or_block=prev_layer_or_block,
            do_batch_norm=True,
        )

        # 2. Grouped 3x3 conv at reduced dimensions
        self.group_conv = _GroupedConv2D(
            blk_name=f"{blockname}/group_conv",
            out_units=self.bottleneck_units,
            cardinality=cardinality,
            strides=strides,
            prev_layer_or_block=self.conv_reduce,
        )

        # 3. 1x1 conv to expand dimensions back
        self.conv_expand = Conv2D1x1(
            name=f"{blockname}/conv_expand",
            units=units,
            activation="linear",
            prev_layer_or_block=self.group_conv,
            do_batch_norm=True,
        )

        self.layers = [self.conv_reduce, self.group_conv, self.conv_expand]

        # Skip connection (1x1 conv if needed) when:
        # stride > 1 (spatial dimensions change) or
        # input channels != output channels
        self.need_projection = (
            strides > 1
            or self._get_input_units(prev_layer_or_block) != units
        )
        if self.need_projection:
            self.skip_branch = Conv2D1x1(
                name=f"{blockname}/skip_conv1x1",
                units=units,
                activation="linear",
                prev_layer_or_block=prev_layer_or_block,
                strides=strides,
                do_batch_norm=True,
            )
            self.layers.append(self.skip_branch)

    # helper --------------------------------------------------------
    def _get_input_units(self, prev_layer_or_block):
        """Helper method to get the number of units/channels from the previous layer."""
        if prev_layer_or_block is None:
            return None
        return getattr(prev_layer_or_block, "units", None)

    # forward -------------------------------------------------------
    def __call__(self, x):
        """Forward pass through the ResNeXt Block."""
        # Main branch
        y = self.conv_reduce(x)
        y = self.group_conv(y)
        y = self.conv_expand(y)

        # Skip connection
        skip = self.skip_branch(x) if self.need_projection else x

        # Sum and apply ReLU activation
        return tf.nn.relu(y + skip)

    # pretty print --------------------------------------------------
    def __str__(self):
        """Custom ResNeXt Block toString method."""
        s = self.blockname + ":"
        for layer in reversed(self.layers[:-1]):
            s += "\n\t" + str(layer)
        if self.need_projection:
            s += "\n\t-->" + str(self.layers[-1]) + "-->"
        return s
