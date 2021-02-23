import tensorflow as tf
from ..layers import *


class UNet(tf.keras.models.Model):
    def __init__(self, filter_sizes=(64, 128, 256, 512), center_filter=1024, output_filter=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filters_sizes = filter_sizes

        self.encoder = [UNetDownConvBlock(i) for i in self.filters_sizes]
        self.center = UNetCenterConvBlock(center_filter)
        self.decoder = [UNetUpConvBlock(i) for i in reversed(self.filters_sizes)]
        self.final_layer = tf.keras.layers.Conv2D(output_filter, (1, 1), activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        skips = []
        for block in self.encoder:
            x, x_s = block(x)
            skips.append(x_s)

        skips.reverse()

        x = self.center(x)

        for i, block in enumerate(self.decoder):
            x = block(x, skips[i])

        return self.final_layer(x)

