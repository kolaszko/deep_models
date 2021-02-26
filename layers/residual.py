import tensorflow as tf


class ResLayer(tf.keras.models.Model):
    '''
    Version of "full pre-activation" residual block proposed in "Identity Mappings in Deep Residual Networks"
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    '''

    def __init__(self, filters, kernel_size, strides=1, dilation=1, padding='same', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.downsample = None

        self.block_1 = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.block_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=strides, padding=padding, dilation_rate=dilation,
                                   kernel_initializer='he_uniform', bias_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=1, padding=padding, dilation_rate=dilation,
                                   kernel_initializer='he_uniform', bias_initializer='he_uniform')

        ])
        if strides != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                       strides=strides, padding=padding, dilation_rate=dilation,
                                       kernel_initializer='he_uniform', bias_initializer='he_uniform'),
            ])

    def call(self, inputs, training=None, mask=None):
        identity = inputs
        x = self.block_1(inputs, training)
        shortcut = self.downsample(x) if self.downsample is not None else identity
        x = self.block_2(x)


        return x + shortcut

