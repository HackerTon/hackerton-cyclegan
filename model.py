import tensorflow as tf
import tensorflow_addons as tfa


def resi_block(input_layer, k):
    # first block
    d_block = tf.keras.layers.Conv2D(
        k, (3, 3), (1, 1), padding='same')(input_layer)
    d_block = tfa.layers.InstanceNormalization()(d_block)
    d_block = tf.keras.layers.Activation('relu')(d_block)

    # second block
    d2_block = tf.keras.layers.Conv2D(
        k, (3, 3), (1, 1), padding='same')(d_block)
    d2_block = tfa.layers.InstanceNormalization()(d_block)
    d2_block = tf.keras.layers.Activation('relu')(d2_block)

    output = tf.keras.layers.Concatenate()([d2_block, input_layer])

    return output


def generator():
    x = tf.keras.layers.Input(shape=(None, None, 3))

    # c7s1-64
    conv1 = tf.keras.layers.Conv2D(64, (7, 7), 1, padding='same')(x)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)

    # d128
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), 2)(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)

    # d256
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), 2)(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)

    # R256 x 9 times
    res = conv3
    for _ in range(9):
        res = resi_block(res, 256)

    # u128
    deconv1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), 2)(res)
    deconv1 = tfa.layers.InstanceNormalization()(deconv1)
    deconv1 = tf.keras.layers.Activation('relu')(deconv1)

    # u64
    deconv2 = tf.keras.layers.Conv2DTranspose(64, (3, 3), 2)(deconv1)
    deconv2 = tfa.layers.InstanceNormalization()(deconv2)
    deconv2 = tf.keras.layers.Activation('relu')(deconv2)

    # c7s1-3
    conv4 = tf.keras.layers.Conv2D(3, (7, 7), 1, padding='same')(deconv2)
    conv4 = tfa.layers.InstanceNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu')(conv4)

    return tf.keras.Model(inputs=[x], outputs=[conv4])
