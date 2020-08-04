import argparse
import datetime
import os
import random
import time

import numpy as np
import tensorflow as tf

import model

BETA_1 = 0.5
AUTOTUNE = tf.data.experimental.AUTOTUNE
LEARNING_RATE = 2e-4

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
                index = random.randint(0, len(self.images) - 1)
                temp = self.images[index]
                self.images[index] = image

                return temp
            else:
                return image


def readdecode(filename, width):
    """
    Can only read JPEG type of file
    """
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (width, width))
    # image = tf.image.random_crop(image, [WIDTH, WIDTH, 3])
    image = image * 2 - 1

    return image


def textparser(text):
    strings = tf.strings.split(text, ' ')
    mask = tf.strings.regex_full_match(strings, '-?1')
    new_strings = tf.boolean_mask(strings, mask)

    link = strings[0]

    return link, new_strings[-20]


def create_dataset_celb(dir, width):
    filepath = os.path.join(dir, 'list_attr_celeba.txt')

    textfile = tf.data.TextLineDataset(filepath)
    textfile = textfile.map(textparser)

    fmale = textfile.filter(lambda _, gender: gender == '-1')
    male = textfile.filter(lambda _, gender: gender == '1')

    adddir = lambda x, y: (dir + 'img_align_celeba/' + x, y)

    fmale = fmale.map(adddir, AUTOTUNE)
    male = male.map(adddir, AUTOTUNE)

    link2image = lambda link, gender: readdecode(link, width)

    fmale = fmale.map(link2image, AUTOTUNE)
    male = male.map(link2image, AUTOTUNE)

    return fmale, male


def train(args):
    date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = f'logs/{date_time}/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    womends, mends = create_dataset_celb(args.d, args.width)

    batch_men = mends.take(args.nsamples).shuffle(args.nsamples).batch(1)
    batch_women = womends.take(args.nsamples).shuffle(args.nsamples).batch(1)

    cyclegan = model.Cyclegan(train_summary_writer, args.lmbda,
                              args.nsamples, args.niters,
                              LEARNING_RATE, BETA_1)
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

    for epoch in range(args.niters):
        init_time = time.time()

        for index, (x, y) in tf.data.Dataset.zip((batch_women, batch_men)).enumerate():
            step = epoch * args.nsamples + index
            g_out, f_out = cyclegan.step_gan(x, y, step)

            g_out = gpool.read(step, g_out)
            f_out = fpool.read(step, f_out)

            cyclegan.step_dis(x, y, g_out, f_out, step)

        final_time = time.time() - init_time

        with train_summary_writer.as_default():
            for test in mends.skip(10).batch(2).take(1):
                image = cyclegan.fgan(test)
                stacked_image = image * 0.5 + 0.5
                tf.summary.image('m from fm', stacked_image, epoch)

        if ((epoch + 1) % 1 == 0):
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
        '--gpu',
        help='enable gpu',
        action='store_true')
    parser.add_argument(
        '--lmbda',
        help='lambda value, default 10',
        default=10)
    parser.add_argument(
        '--nsamples',
        help='number of samples to be taken from database',
        default=1000)
    parser.add_argument(
        '--niters',
        help='number of iterations to run',
        default=200
    )
    parser.add_argument(
        '--width',
        help='width of the image to be trained on',
        default=256
    )

    args = parser.parse_args()
    train(args)
