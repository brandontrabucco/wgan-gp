from wgangp.generator import Generator
from wgangp.discriminator import Discriminator
from wgangp.utils import discriminator_loss, generator_loss, get_p_yx, inception_score
import tensorflow as tf
import tensorflow_datasets as tfds


if __name__ == '__main__':

    dataset = tfds.load('cifar10', split='train', shuffle_files=True)
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    c = tf.keras.applications.inception_v3.InceptionV3()

    g = Generator(128, 128, 3)
    g_optim = tf.keras.optimizers.Adam(learning_rate=0.0002,
                                       beta_1=0.0,
                                       beta_2=0.9)

    d = Discriminator(128, 128, 3)
    d_optim = tf.keras.optimizers.Adam(learning_rate=0.0002,
                                       beta_1=0.0,
                                       beta_2=0.9)

    out = d(g(tf.random.normal([1, 128])))

    iteration = 0

    g.save('generator.h5')
    d.save('discriminator.h5')

    for epoch in range(1000):

        for batch in dataset:

            images = batch['image']
            images = tf.cast(images, tf.float32) / 127.5 - 1.0
            batch_size = tf.shape(images)[0]

            if iteration % 100 == 0:

                p_yx = tf.concat([get_p_yx(g(tf.random.normal([100, 128])), c)
                                  for i in range(50)], 0)

                print("Iteration {} Inception Score {}".format(
                    iteration, inception_score(p_yx)))

            def d_loss_function():
                latent_var = tf.random.normal([batch_size, 128])
                return discriminator_loss(g, d, images, latent_var, training=True)

            d_optim.minimize(d_loss_function, d.trainable_variables)

            if iteration % 5 == 0:

                def g_loss_function():
                    latent_var = tf.random.normal([batch_size, 128])
                    return generator_loss(g, d, latent_var, training=True)

                g_optim.minimize(g_loss_function, g.trainable_variables)

            iteration += 1

        g.save('generator.h5')
        d.save('discriminator.h5')
