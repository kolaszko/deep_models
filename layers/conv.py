import tensorflow as tf


class UNetDownConvBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D((2, 2), strides=2)

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.conv_2(x)

        return self.max_pool(x), x


class UNetUpConvBlock(tf.keras.models.Model):
    '''
    Instead of UpSampling2d used Conv2Transpose
    '''
    def __init__(self, filters, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters, 3, strides=(2, 2), activation='relu', padding='same')
        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')

    def call(self, inputs, skip, training=None, mask=None):
        x = self.conv_transpose(inputs)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class UNetCenterConvBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        return self.conv_2(x)
