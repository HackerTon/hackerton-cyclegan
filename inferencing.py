import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def convert():
    g_gan = pix2pix.unet_generator(3, norm_type='instancenorm')
    f_gan = pix2pix.unet_generator(3, norm_type='instancenorm')

    dis_g = pix2pix.discriminator(
        norm_type='instancenorm', target=False)
    dis_f = pix2pix.discriminator(
        norm_type='instancenorm', target=False)

    ckpt = tf.train.Checkpoint(
        g_gan=g_gan,
        f_gan=f_gan,
        dis_g=dis_g,
        dis_f=dis_f
    )

    if tf.train.latest_checkpoint('checkpoints'):

        manager = tf.train.CheckpointManager(ckpt, 'checkpoints', 5)

        for index, chpt in enumerate(manager.checkpoints):
            print(f'{index} : {chpt}')

        index = int(input('Index'))

        print('loaded checkpoint:', manager.checkpoints[index])
        ckpt.restore(manager.checkpoints[index])

    g_gan.save('./g_gan')
    f_gan.save('./f_gan')


if __name__ == "__main__":
    convert()
