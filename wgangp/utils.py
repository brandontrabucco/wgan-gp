import tensorflow as tf


def gradient_penalty(discriminator, inputs, **kwargs):
    """Evaluates a gradient penalty term that penalizes deviations
    of the gradient of a model from unit norm

    Arguments:

    discriminator: tf.keras.Model
        a keras model that processes images and returns probabilities
        that a sample is real or fake
    inputs: tf.Tensor
        the center to evaluate the gradient penalty of a model at
        an image with shape [batch, height, width, 3]

    Returns:

    penalty: tf.Tensor
        penalty of deviation of gradients of the discriminator from
        unit norm with shape [batch]"""

    with tf.GradientTape() as tape:

        tape.watch(inputs)

        outputs = discriminator(inputs, **kwargs)

    grad = tape.gradient(outputs, inputs)

    grad = tf.reshape(grad, [tf.shape(inputs)[0], -1])

    return (tf.linalg.norm(grad, axis=1, ord=2) - 1.) ** 2


def discriminator_loss(g, d, images, latent_var, **kwargs):
    """Evaluates a gradient penalty term that penalizes deviations
    of the gradient of a model from unit norm

    Arguments:

    generator: tf.keras.Model
        a keras model that processes latent variables and returns images
        that that fool a discriminator
    discriminator: tf.keras.Model
        a keras model that processes images and returns probabilities
        that a sample is real or fake
    images: tf.Tensor
        real sample images from a dataset that will be used to train the
        discriminator with shape [batch, height, width, 3]
    latent_var: tf.Tensor
        latent variables that provide stochasticity to the generator
        and have shape [batch, channels]

    Returns:

    loss: tf.Tensor
        the value of the loss function that will be used to train the
        discriminator and has shape []"""

    x = g(latent_var, **kwargs)

    e = tf.random.uniform([tf.shape(images)[0], 1, 1, 1])

    xp = e * images + (1 - e) * x

    return tf.reduce_mean(d(x, **kwargs) -
                          d(images, **kwargs) +
                          10 * gradient_penalty(d, xp, **kwargs))


def generator_loss(g, d, latent_var, **kwargs):
    """Evaluates a gradient penalty term that penalizes deviations
    of the gradient of a model from unit norm

    Arguments:

    generator: tf.keras.Model
        a keras model that processes latent variables and returns images
        that that fool a discriminator
    discriminator: tf.keras.Model
        a keras model that processes images and returns probabilities
        that a sample is real or fake
    latent_var: tf.Tensor
        latent variables that provide stochasticity to the generator
        and have shape [batch, channels]

    Returns:

    loss: tf.Tensor
        the value of the loss function that will be used to train the
        discriminator and has shape []"""

    x = g(latent_var, **kwargs)

    return -tf.reduce_mean(d(x, **kwargs))


def get_p_yx(images, classifier):
    """Evaluates a gradient penalty term that penalizes deviations
    of the gradient of a model from unit norm

    Arguments:

    images: tf.Tensor
        a tensor containing images sampled from a generative model and scaled
        in the range [-1, 1] and has shape [batch, height, width, 3]
    classifier: tf.keras.Model
        a keras model that is used to calculate the class scores of
        images generates using a generative model

    Returns:

    loss: tf.Tensor
        the value of the loss function that will be used to train the
        discriminator and has shape []"""

    images = tf.image.resize(images, [299, 299])

    images = tf.clip_by_value(images * 127.5 + 127.5, 0.0, 255.5)

    return classifier.predict(
        tf.keras.applications.inception_v3.preprocess_input(images))


def inception_score(p_yx):
    """Evaluates a gradient penalty term that penalizes deviations
    of the gradient of a model from unit norm

    Arguments:

    images: tf.Tensor
        a tensor containing images sampled from a generative model and scaled
        in the range [-1, 1] and has shape [batch, height, width, 3]
    classifier: tf.keras.Model
        a keras model that is used to calculate the class scores of
        images generates using a generative model
    n_split: int
        the number of batches to split the images tensor into; used
        for calculating inception scores of large batches

    Returns:

    loss: tf.Tensor
        the value of the loss function that will be used to train the
        discriminator and has shape []"""

    p_y = tf.expand_dims(tf.reduce_mean(p_yx, axis=0), 0)

    kl_d = p_yx * (tf.math.log(p_yx + 1e-9) - tf.math.log(p_y + 1e-9))

    log_is = tf.reduce_mean(tf.reduce_sum(kl_d, axis=1))

    return tf.math.exp(log_is)
