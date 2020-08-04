import tensorflow as tf
import scheduler
import tensorflow_addons as tfa
from tensorflow_examples.models.pix2pix import pix2pix


def resi_block(input_layer, k):
    kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02)

    # first block
    d_block = tf.keras.layers.Conv2D(
        k, (3, 3), (1, 1), padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    d_block = tfa.layers.InstanceNormalization()(d_block)
    d_block = tf.keras.layers.Activation('relu')(d_block)

    # second block
    d2_block = tf.keras.layers.Conv2D(
        k, (3, 3), (1, 1), padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(d_block)
    d2_block = tfa.layers.InstanceNormalization()(d_block)
    d2_block = tf.keras.layers.Activation('relu')(d2_block)

    output = tf.keras.layers.Add()([d2_block, input_layer])

    return output


def generator():
    kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02)

    x = tf.keras.layers.Input(shape=(None, None, 3))

    # c7s1-64
    conv1 = tf.keras.layers.Conv2D(
        64, (7, 7), 1, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)

    # d128
    conv2 = tf.keras.layers.Conv2D(
        128, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)

    # d256
    conv3 = tf.keras.layers.Conv2D(
        256, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)

    # R256 x 9 times
    res = conv3
    for _ in range(9):
        res = resi_block(res, 256)

    # u128
    deconv1 = tf.keras.layers.Conv2DTranspose(
        128, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(res)
    deconv1 = tfa.layers.InstanceNormalization()(deconv1)
    deconv1 = tf.keras.layers.Activation('relu')(deconv1)

    # u64
    deconv2 = tf.keras.layers.Conv2DTranspose(
        64, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(deconv1)
    deconv2 = tfa.layers.InstanceNormalization()(deconv2)
    deconv2 = tf.keras.layers.Activation('relu')(deconv2)

    # c7s1-3
    conv4 = tf.keras.layers.Conv2D(
        3, (7, 7), 1, padding='same', use_bias=False)(deconv2)
    conv4 = tfa.layers.InstanceNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('tanh')(conv4)

    return tf.keras.Model(inputs=[x], outputs=[conv4])


def disloss(distruth, disfake):
    loss1 = tf.reduce_mean(tf.square(distruth - tf.ones_like(distruth)))
    loss2 = tf.reduce_mean(tf.square(disfake))

    return (loss1 + loss2) * 0.5


def identity_loss(fakeimage, realimage):
    loss = tf.reduce_mean(tf.abs(fakeimage - realimage))
    return 0.5 * loss


def ganloss(disfake):
    return tf.reduce_mean(tf.square(disfake - tf.ones_like(disfake)))


def cycleloss(fakex, fakey, realx, realy):
    loss1 = tf.reduce_mean(tf.abs(fakex - realx))
    loss2 = tf.reduce_mean(tf.abs(fakey - realy))

    return loss1 + loss2


class Cyclegan:
    def __init__(self, summary, lmbda, nsamples, niters, learning_rate, beta_1):
        self.lmbda = tf.constant(lmbda, tf.float32)
        self.summary = summary
        self.ggan = generator()
        self.fgan = generator()

        self.gdis = pix2pix.discriminator(
            norm_type='instancenorm', target=False)
        self.fdis = pix2pix.discriminator(
            norm_type='instancenorm', target=False)

        self.lrscheculer = scheduler.LinearDecay(
            nsamples * niters // 2, learning_rate, nsamples * niters // 2, 0.0)

        self.opti_ggan = tf.keras.optimizers.Adam(self.lrscheculer, beta_1)
        self.opti_gdis = tf.keras.optimizers.Adam(self.lrscheculer, beta_1)
        self.opti_fgan = tf.keras.optimizers.Adam(self.lrscheculer, beta_1)
        self.opti_fdis = tf.keras.optimizers.Adam(self.lrscheculer, beta_1)

    @tf.function
    def step_dis(self, x, y, xrec, yrec, step):
        with self.summary.as_default():
            with tf.GradientTape(persistent=True) as tape:
                disgtruth = self.gdis(y)
                disftruth = self.fdis(x)
                disgfake = self.gdis(xrec)
                disffake = self.fdis(yrec)

                disgloss = disloss(disgtruth, disgfake)
                disfloss = disloss(disftruth, disffake)

            tf.summary.scalar('dgloss', disgloss, step)
            tf.summary.scalar('dfloss', disfloss, step)

        disggrad = tape.gradient(disgloss, self.gdis.trainable_variables)
        disfgrad = tape.gradient(disfloss, self.fdis.trainable_variables)

        self.opti_gdis.apply_gradients(
            zip(disggrad, self.gdis.trainable_variables))
        self.opti_fdis.apply_gradients(
            zip(disfgrad, self.fdis.trainable_variables))

    @tf.function
    def step_gan(self, x, y, step):
        with self.summary.as_default():
            with tf.GradientTape(persistent=True) as tape:
                fakey = self.ggan(x)
                cfakex = self.fgan(fakey)
                fakex = self.fgan(y)
                cfakey = self.ggan(fakex)

                disfakey = self.gdis(fakey)
                disfakex = self.fdis(fakex)

                cyloss = self.lmbda * cycleloss(cfakex, cfakey, x, y)
                gloss = ganloss(disfakey)
                floss = ganloss(disfakex)

                # disgfakeloss = tf.reduce_mean(tf.pow(dis_g_out, 2))
                # disffakeloss = tf.reduce_mean(tf.pow(dis_f_out, 2))

                # g_y = self.ggan(y, training=True)
                # f_x = self.fgan(x, training=True)

                # identity_loss_g = identity_loss(g_y, y) * self.lmbda
                # identity_loss_f = identity_loss(f_x, x) * self.lmbda

                totalgloss = gloss + cyloss
                totalfloss = floss + cyloss

            tf.summary.scalar('gloss', gloss, step)
            tf.summary.scalar('floss', floss, step)
            tf.summary.scalar('cyloss', cyloss, step)

        g_grads = tape.gradient(totalgloss, self.ggan.trainable_variables)
        f_grads = tape.gradient(totalfloss, self.fgan.trainable_variables)

        self.opti_ggan.apply_gradients(
            zip(g_grads, self.ggan.trainable_variables))
        self.opti_fgan.apply_gradients(
            zip(f_grads, self.fgan.trainable_variables))

        return fakey, fakex
