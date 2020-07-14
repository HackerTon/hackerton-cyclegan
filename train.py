import argparse
import datetime
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import python
from tensorflow_examples.models.pix2pix import pix2pix

import model
import scheduler

NUM_ITERATION = 200
LAMBDA = tf.constant(5, tf.float32)
TAKEN_NUM = 1000
BETA_1 = 0.5
AUTOTUNE = tf.data.experimental.AUTOTUNE
LEARNING_RATE = 2e-4
WIDTH = 256

tf.random.set_seed(123456)


class Pool:
    def __init__(self, size=50):
        self.size = size
        self.images = []

    def read(self, step, image):
        """
        read images with random sampling
        return image unchanged if step < size=50
        """

        if step < self.size:
            self.images.append(image)
            return image
        else:
            if random.randint(0, 1):
                index = random.randint(0, self.size - 1)
                temp = self.images[index]
                self.images[index] = image

                return temp
            else:
                return image


def readdecode(filename):
    """
    Can only read JPEG type of file
    """
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (WIDTH + 10, WIDTH + 10))
    image = tf.image.random_crop(image, [WIDTH, WIDTH, 3])
    image = image * 2 - 1

    return image


def disloss(dis_y, dis_g_x):
    loss1 = tf.reduce_mean(tf.pow(dis_y - tf.ones_like(dis_y), 2))
    loss2 = tf.reduce_mean(tf.pow(dis_g_x, 2))

    return (loss1 + loss2) * 0.5


def identity_loss(gen_image, image):
    loss = tf.reduce_mean(tf.abs(gen_image - image))
    return LAMBDA * 0.5 * loss


def ganloss(d_g_x):
    return tf.reduce_mean(tf.pow(d_g_x - tf.ones_like(d_g_x), 2))


def cycleloss(f_g_x, g_f_x, x, y):
    loss1 = tf.reduce_mean(tf.abs(f_g_x - x))
    loss2 = tf.reduce_mean(tf.abs(g_f_x - y))

    return loss1 + loss2


def textparser(text):
    strings = tf.strings.split(text, ' ')
    mask = tf.strings.regex_full_match(strings, '-?1')
    new_strings = tf.boolean_mask(strings, mask)

    link = strings[0]

    return link, strings[-20]


# BECARE NOT UPDATED
def create_dataset():
    mends = tf.data.Dataset.list_files(
        '/home/hackerton/Downloads/menwomen/men/*.jpg')
    womends = tf.data.Dataset.list_files(
        '/home/hackerton/Downloads/menwomen/women/*.jpg')

    mends = mends.map(readdecode, AUTOTUNE)
    womends = womends.map(readdecode, AUTOTUNE)

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

    fmale = fmale.map(adddir, AUTOTUNE)
    male = male.map(adddir, AUTOTUNE)

    fmale = fmale.map(link2image, AUTOTUNE)
    male = male.map(link2image, AUTOTUNE)

    return fmale, male


def testdataset(args):
    fmale, male = create_dataset_celb(args.d)

    import matplotlib.pyplot as plt

    for i in fmale.take(1):
        print(i)

        fig = plt.figure()
        plt.imshow(i)
        plt.savefig('image.png')


class Cyclegan:
    def __init__(self, summary):
        self.summary = summary
        self.ggan = model.generator()
        self.fgan = model.generator()

        self.gdis = pix2pix.discriminator(
            norm_type='instancenorm', target=False)
        self.fdis = pix2pix.discriminator(
            norm_type='instancenorm', target=False)

        self.lrscheculer = scheduler.LinearDecay(
            TAKEN_NUM * NUM_ITERATION // 2, LEARNING_RATE, TAKEN_NUM * NUM_ITERATION // 2, 0.0)

        self.opti_ggan = tf.keras.optimizers.Adam(self.lrscheculer, BETA_1)
        self.opti_fgan = tf.keras.optimizers.Adam(self.lrscheculer, BETA_1)
        self.opti_gdis = tf.keras.optimizers.Adam(self.lrscheculer, BETA_1)
        self.opti_fdis = tf.keras.optimizers.Adam(self.lrscheculer, BETA_1)

    @tf.function
    def step_dis(self, x, y, xrec, yrec, step):
        with self.summary.as_default():
            with tf.GradientTape(persistent=True) as tape:
                disgtruth = self.gdis(y, training=True)
                disftruth = self.fdis(x, training=True)
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
                g_out = self.ggan(x, training=True)
                cycle_g = self.fgan(g_out, training=True)
                f_out = self.fgan(y, training=True)
                cycle_f = self.ggan(f_out, training=True)

                dis_g_out = self.gdis(g_out, training=True)
                dis_f_out = self.fdis(f_out, training=True)

                cyloss = LAMBDA * cycleloss(cycle_g, cycle_f, x, y)
                g_loss = ganloss(dis_g_out)
                f_loss = ganloss(dis_f_out)

                disgfakeloss = tf.reduce_mean(tf.pow(dis_g_out, 2))
                disffakeloss = tf.reduce_mean(tf.pow(dis_f_out, 2))

                # identity_loss_g = identity_loss(g_y, y)
                # identity_loss_f = identity_loss(f_x, x)

                total_gan_loss_g = g_loss + cyloss
                total_gan_loss_f = f_loss + cyloss

            tf.summary.scalar('gloss', g_loss, step)
            tf.summary.scalar('floss', f_loss, step)
            tf.summary.scalar('cyloss', cyloss, step)

        g_grads = tape.gradient(g_loss, self.ggan.trainable_variables)
        f_grads = tape.gradient(f_loss, self.fgan.trainable_variables)

        self.opti_ggan.apply_gradients(
            zip(g_grads, self.ggan.trainable_variables))
        self.opti_fgan.apply_gradients(
            zip(f_grads, self.fgan.trainable_variables))

        return g_out, f_out


def train(args):
    if args.gpu is None:
        print('CPU NOT SUPPORTED')
        return

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'logs/{date_time}/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    womends, mends = create_dataset_celb(args.d)

    batch_men = mends.take(TAKEN_NUM).shuffle(TAKEN_NUM).batch(1)
    batch_women = womends.take(TAKEN_NUM).shuffle(TAKEN_NUM).batch(1)

    cyclegan = Cyclegan(train_summary_writer)
    gpool = Pool()
    fpool = Pool()

    checkpoint_path = './checkpoints/'
    ckpt = tf.train.Checkpoint(
        ggan=cyclegan.ggan,
        fgan=cyclegan.fgan,
        gdis=cyclegan.gdis,
        fdis=cyclegan.fdis,
        opti_ggan=cyclegan.opti_ggan,
        opti_fgan=cyclegan.opti_fgan,
        opti_gdis=cyclegan.opti_gdis,
        opti_fdis=cyclegan.opti_fdis
    )

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f'Loaded previous checkpoint {ckpt_manager.latest_checkpoint}')

    for epoch in range(NUM_ITERATION):
        init_time = time.time()

        for index, (x, y) in tf.data.Dataset.zip((batch_women, batch_men)).enumerate():
            step = epoch * TAKEN_NUM + index
            g_out, f_out = cyclegan.step_gan(x, y, step)

            g_out = gpool.read(step, g_out)
            f_out = fpool.read(step, f_out)

            cyclegan.step_dis(x, y, g_out, f_out, step)

        final_time = time.time() - init_time

        with train_summary_writer.as_default():
            for test in mends.skip(TAKEN_NUM + 1).take(1):
                image = cyclegan.fgan(tf.expand_dims(test, 0))
                stacked_image = tf.stack([image[0], test]) * 0.5 + 0.5
                tf.summary.image('m from fm', stacked_image, epoch)

        if ((epoch+1) % 1 == 0):
            ckpt_manager.save()
            print(f'save checkpoint at {epoch + 1}')

        print(f'epoch: {epoch+1} took {round(final_time)} \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', required=True,
        help="""directory that container imgtextfile and
         img_align_celeba(must end with "/")""")
    parser.add_argument(
        '--gpu', help='enable gpu', action='store_true')
    args = parser.parse_args()

    train(args)
