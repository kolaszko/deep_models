import tensorflow as tf

from models import ERFNet, UNet


if __name__ == '__main__':

    a = UNet()

    t = tf.random.uniform(shape=(1, 512, 512, 3))

    o = a(t)
    print(o.shape)