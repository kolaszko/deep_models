import tensorflow as tf

from models import ERFNet, UNet


if __name__ == '__main__':
    a = UNet()

    t = tf.random.uniform(shape=(4, 192, 256, 3))

    output = a(t)
    print(output.shape)
