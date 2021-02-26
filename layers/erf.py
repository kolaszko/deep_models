import tensorflow as tf


class DownSample(tf.keras.models.Model):
        def __init__(self, in_filters, out_filters, kernel_size, stride=2, momentum=0.99, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conv = tf.keras.layers.Conv2D(out_filters - in_filters, kernel_size, stride, padding='same',
                                               kernel_initializer='he_uniform', bias_initializer='he_uniform')
            self.max_pool = tf.keras.layers.MaxPool2D(stride, stride, padding='same')
            self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
            self.activation = tf.keras.layers.ReLU()

        def call(self, inputs, training=None, mask=None):
            a = self.conv(inputs)
            b = self.max_pool(inputs)

            x = tf.concat([a, b], -1)
            x = self.bn(x, training=training)

            return self.activation(x)


class UpSample(tf.keras.models.Model):
    def __init__(self, filters, kernel_size, stride=2, momentum=0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, stride, padding='same',
                                                    kernel_initializer='he_uniform', bias_initializer='he_uniform')
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)

        return self.activation(x)
