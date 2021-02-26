import tensorflow as tf

from layers import ResLayer


class ResNet18BackBone(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same',
                                             kernel_initializer='he_uniform', bias_initializer='he_uniform')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.res = tf.keras.Sequential([
            ResLayer(64, 3, 1),
            ResLayer(64, 3, 1),
            ResLayer(128, 3, 2),
            ResLayer(128, 3, 1),
            ResLayer(256, 3, 2),
            ResLayer(256, 3, 1),
            ResLayer(512, 3, 2),
            ResLayer(512, 3, 1),
            ])


    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training)
        x = self.pool_1(x)

        return self.res(x, training)