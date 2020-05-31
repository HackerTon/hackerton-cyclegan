import tensorflow as tf
import numpy as np
import datetime
import time
import argparse
import os
from tensorflow_examples.models.pix2pix import pix2pix

NUM_ITERATION = 200
LAMBDA = tf.constant(10, tf.float16)
TAKEN_NUM = 100


def readdecode(filename):
    """
    Can only read JPEG type of file
    """
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = image * 2 - 1
    image = tf.image.convert_image_dtype(image, tf.float16)

    return image


def identity_loss(gen_image, image):
    loss = tf.reduce_mean(tf.abs(gen_image - image))

    return LAMBDA * 0.5 * loss


def ganloss(d_g_x):
    return tf.reduce_mean(tf.pow(d_g_x - tf.ones_like(d_g_x), 2))


def disloss(dis_y, dis_g_x):
    loss1 = tf.reduce_mean(tf.pow(dis_y - tf.ones_like(dis_y), 2))
    loss2 = tf.reduce_mean(tf.pow(dis_g_x, 2))

    return (loss1 + loss2) * 0.5


def cycleloss(f_g_x, g_f_x, x, y):
    loss1 = tf.reduce_mean(tf.abs(f_g_x - x))
    loss2 = tf.reduce_mean(tf.abs(g_f_x - y))

    return loss1 + loss2


def textparser(text):
    strings = tf.strings.split(text, ' ')
    mask = tf.strings.regex_full_match(strings, '-?1')
    new_strings = tf.boolean_mask(strings, mask)

    link = strings[0]

    return link, new_strings[-20]


# BECARE NOT UPDATED
def create_dataset():
    mends = tf.data.Dataset.list_files(
        '/home/hackerton/Downloads/menwomen/men/*.jpg')
    womends = tf.data.Dataset.list_files(
        '/home/hackerton/Downloads/menwomen/women/*.jpg')

    mends = mends.map(readdecode)
    womends = womends.map(readdecode)

    mends = mends.take(TAKEN_NUM)
    womends = womends.take(TAKEN_NUM)

    return mends, womends


def link2image(link, gender):
    img = readdecode(link)

    return img


def create_dataset_celb(dir):
    filepath = os.path.join(dir, 'list_attr_celeba.txt')

    textfile = tf.data.TextLineDataset(filepath)
    textfile = textfile.map(textparser)

    fmale = textfile.filter(lambda link, gender: gender == '-1')
    male = textfile.filter(lambda link, gender: gender == '1')

    def adddir(x, y):
        return dir + 'img_align_celeba/' + x, y

    fmale = fmale.map(adddir)
    male = male.map(adddir)

    fmale = fmale.map(link2image, tf.data.experimental.AUTOTUNE)
    male = male.map(link2image, tf.data.experimental.AUTOTUNE)

    fmale = fmale.shuffle(100)
    male = male.shuffle(100)

    return fmale, male


def testdataset(args):
    fmale, male = create_dataset_celb(args.d)

    import matplotlib.pyplot as plt

    for i in fmale.take(1):
        print(i)

        fig = plt.figure()
        plt.imshow(i)
        plt.savefig('image.png')


def train(args):
    if args.gpu is None:
        print('CPU NOT SUPPORTED')
        return

    @tf.function
    def train_step(x, y):
        with tf.GradientTape(persistent=True) as tape:
            g_out = g_gan(x, training=True)
            cycle_g = f_gan(g_out, training=True)

            f_out = f_gan(y, training=True)
            cycle_f = g_gan(f_out, training=True)

            cyloss = cycleloss(cycle_g, cycle_f, x, y)

            dis_g_out = dis_g(g_out, training=True)
            dis_f_out = dis_f(f_out, training=True)
            dis_g_y = dis_g(y, training=True)
            dis_f_y = dis_f(x, training=True)

            g_loss = ganloss(dis_g_out)
            f_loss = ganloss(dis_f_out)

            dis_g_loss = disloss(dis_g_y, dis_g_out)
            dis_f_loss = disloss(dis_f_y, dis_f_out)

            g_y = g_gan(y, training=True)
            f_x = f_gan(x, training=True)

            total_gan_loss_g = g_loss + (LAMBDA*cyloss) + identity_loss(g_y, y)
            total_gan_loss_f = f_loss + (LAMBDA*cyloss) + identity_loss(f_x, x)

        g_grads = tape.gradient(
            total_gan_loss_g, g_gan.trainable_variables)
        f_grads = tape.gradient(
            total_gan_loss_f, f_gan.trainable_variables)

        g_gan_opti.apply_gradients(
            zip(g_grads, g_gan.trainable_variables))
        f_gan_opti.apply_gradients(
            zip(f_grads, f_gan.trainable_variables))

        dis_g_grads = tape.gradient(
            dis_g_loss, dis_g.trainable_variables)
        dis_f_grads = tape.gradient(
            dis_f_loss, dis_f.trainable_variables)

        dis_g_opti.apply_gradients(
            zip(dis_g_grads, dis_g.trainable_variables))
        dis_f_opti.apply_gradients(
            zip(dis_f_grads, dis_f.trainable_variables))

        return g_loss, f_loss, dis_g_loss, dis_f_loss, g_out, f_out

    @tf.function
    def train_step_g(x, y):
        with tf.GradientTape(persistent=True) as tape:
            g_out = g_gan(x, training=True)
            cycle_g = f_gan(g_out, training=True)
            f_out = f_gan(y, training=True)
            cycle_f = g_gan(f_out, training=True)

            dis_g_out = dis_g(g_out, training=True)
            dis_f_out = dis_f(f_out, training=True)
            dis_g_y = dis_g(y, training=True)
            dis_f_y = dis_f(x, training=True)

            dis_g_loss = disloss(dis_g_y, dis_g_out)
            dis_f_loss = disloss(dis_f_y, dis_f_out)

            cyloss = cycleloss(cycle_g, cycle_f, x, y)

            g_loss = ganloss(dis_g_out)
            f_loss = ganloss(dis_f_out)

            g_y = g_gan(y, training=True)
            f_x = f_gan(x, training=True)

            total_gan_loss_g = g_loss + (LAMBDA*cyloss) + identity_loss(g_y, y)
            total_gan_loss_f = f_loss + (LAMBDA*cyloss) + identity_loss(f_x, x)

            # Scale Gradient
            scaled_loss_g = g_gan_opti.get_scaled_loss(total_gan_loss_g)
            scaled_loss_f = f_gan_opti.get_scaled_loss(total_gan_loss_f)
            scaled_loss_dg = dis_g_opti.get_scaled_loss(dis_g_loss)
            scaled_loss_df = dis_f_opti.get_scaled_loss(dis_f_loss)

        # Calculate scale gradient
        g_grads = tape.gradient(scaled_loss_g, g_gan.trainable_variables)
        f_grads = tape.gradient(scaled_loss_f, f_gan.trainable_variables)
        dis_g_grads = tape.gradient(scaled_loss_dg, dis_g.trainable_variables)
        dis_f_grads = tape.gradient(scaled_loss_df, dis_f.trainable_variables)

        # Unscale gradient
        g_grads = g_gan_opti.get_unscaled_gradients(g_grads)
        f_grads = f_gan_opti.get_unscaled_gradients(f_grads)
        dis_g_grads = dis_g_opti.get_unscaled_gradients(dis_g_grads)
        dis_f_grads = dis_f_opti.get_unscaled_gradients(dis_f_grads)

        # Apply unscaled gradient
        g_gan_opti.apply_gradients(zip(g_grads, g_gan.trainable_variables))
        f_gan_opti.apply_gradients(zip(f_grads, f_gan.trainable_variables))
        dis_g_opti.apply_gradients(zip(dis_g_grads, dis_g.trainable_variables))
        dis_f_opti.apply_gradients(zip(dis_f_grads, dis_f.trainable_variables))

        return g_loss, f_loss, dis_g_loss, dis_f_loss

    if args.gpu:
        print('GPU MODE')
        from tensorflow.keras import mixed_precision
        mixed_precision.experimental.set_policy('float16')

        g_gan = pix2pix.unet_generator(3, norm_type='instancenorm')
        f_gan = pix2pix.unet_generator(3, norm_type='instancenorm')

        dis_g = pix2pix.discriminator(
            norm_type='instancenorm', target=False)
        dis_f = pix2pix.discriminator(
            norm_type='instancenorm', target=False)

        dis_g.layers.append(tf.keras.layers.Activation('sigmoid'))
        dis_f.layers.append(tf.keras.layers.Activation('sigmoid'))

        g_gan_opti = tf.keras.optimizers.Adam(2e-04)
        f_gan_opti = tf.keras.optimizers.Adam(2e-04)
        dis_g_opti = tf.keras.optimizers.Adam(2e-04)
        dis_f_opti = tf.keras.optimizers.Adam(2e-04)

        g_gan_opti = mixed_precision.experimental.LossScaleOptimizer(
            g_gan_opti, 'dynamic')
        f_gan_opti = mixed_precision.experimental.LossScaleOptimizer(
            f_gan_opti, 'dynamic')
        dis_g_opti = mixed_precision.experimental.LossScaleOptimizer(
            dis_g_opti, 'dynamic')
        dis_f_opti = mixed_precision.experimental.LossScaleOptimizer(
            dis_f_opti, 'dynamic')
    else:
        print('CPU MODE')
        g_gan = pix2pix.unet_generator(3, norm_type='instancenorm')
        f_gan = pix2pix.unet_generator(3, norm_type='instancenorm')

        dis_g = pix2pix.discriminator(
            norm_type='instancenorm', target=False)
        dis_f = pix2pix.discriminator(
            norm_type='instancenorm', target=False)

        dis_g.layers.append(tf.keras.layers.Activation('sigmoid'))
        dis_f.layers.append(tf.keras.layers.Activation('sigmoid'))

        g_gan_opti = tf.keras.optimizers.Adam(2e-04)
        f_gan_opti = tf.keras.optimizers.Adam(2e-04)

        dis_g_opti = tf.keras.optimizers.Adam(2e-04)
        dis_f_opti = tf.keras.optimizers.Adam(2e-04)

    womends, mends = create_dataset_celb(args.d)

    loss_name = ['g_loss', 'f_loss', 'dis_g_loss', 'dis_f_loss']

    batch_men = mends.take(TAKEN_NUM).batch(1)
    batch_women = womends.take(TAKEN_NUM).batch(1)

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'logs/{date_time}/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    checkpoint_path = './checkpoints/'
    ckpt = tf.train.Checkpoint(
        g_gan=g_gan,
        f_gan=f_gan,
        dis_g=dis_g,
        dis_f=dis_f,
        g_gan_opti=g_gan_opti,
        f_gan_opti=f_gan_opti,
        dis_g_opti=dis_g_opti,
        dis_f_opti=dis_f_opti
    )

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f'Loaded previous checkpoint {ckpt_manager.latest_checkpoint}')

    for epoch in range(NUM_ITERATION):
        history = np.zeros([100, 4])

        init_time = time.time()

        if args.gpu:
            for index, (x, y) in tf.data.Dataset.zip((batch_women, batch_men)).enumerate():
                # g_loss, f_loss, dis_g_loss, dis_f_loss = train_step_g(x, y)
                train_step_g(x, y)

                # history[index] = [g_loss, f_loss, dis_g_loss, dis_f_loss]
        else:
            with tf.device('/cpu:0'):
                for index, (x, y) in tf.data.Dataset.zip((batch_women, batch_men)).enumerate():
                    g_loss, f_loss, dis_g_loss, dis_f_loss = train_step(x, y)

                    history[index] = [g_loss, f_loss, dis_g_loss, dis_f_loss]

        final_time = time.time() - init_time

        print(f'epoch: {epoch+1} took {round(final_time)}')

        with train_summary_writer.as_default():
            for loss, name in zip(np.mean(history, axis=0), loss_name):
                tf.summary.scalar(name, loss, epoch)
                print(f'{name}: {loss}')

            tf.summary.image('output_fm_from_m', f_gan((y + 1) * 0.5), epoch)

        # ckpt_manager.save()

        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', required=True,
        help="""directory that container imgtextfile and
         img_align_celeba(must end with "/")""")
    parser.add_argument('--gpu', help='enable low memory mode')
    args = parser.parse_args()

    train(args)
