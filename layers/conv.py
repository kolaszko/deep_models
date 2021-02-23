import tensorflow as tf


class UNetDownConvBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.2)

        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.dropout_2 = tf.keras.layers.Dropout(rate=0.2)
        self.max_pool = tf.keras.layers.MaxPool2D((2, 2), strides=2)

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.dropout_1(x, training=training)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.dropout_2(x, training=training)

        return self.max_pool(x), x


class UNetUpConvBlock(tf.keras.models.Model):
    '''
    Instead of UpSampling2d used Conv2Transpose
    '''

    def __init__(self, filters, kernel_size=3, dropout_rate=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters, 3, strides=(2, 2), activation='relu',
                                                              padding='same')
        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, skip, training=None, mask=None):
        x = self.conv_transpose(inputs)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = self.dropout_1(x, training=training)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.dropout_2(x, training=training)
        return x


class UNetCenterConvBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, dropout_rate=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.dropout_1(x, training=training)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.dropout_2(x, training=training)
        return x
