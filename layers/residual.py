import tensorflow as tf


class ResLayer(tf.keras.models.Model):
    '''
    Version of "full pre-activation" residual block proposed in "Identity Mappings in Deep Residual Networks"
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    '''

    def __init__(self, filters, kernel_size, strides=1, dilation=1, dropout=0.2, padding='same', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=strides, padding=padding, dilation_rate=dilation,
                                   kernel_initializer='he_uniform', bias_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=strides, padding=padding, dilation_rate=dilation,
                                   kernel_initializer='he_uniform', bias_initializer='he_uniform')

        ])

    def call(self, inputs, training=None, mask=None):
        x = self.block(inputs)
        return x + inputs

