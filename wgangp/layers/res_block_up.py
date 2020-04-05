import tensorflow as tf


class ResBlockUp(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size,
                 output_size,
                 **kwargs):
        """Creates a 'network-in-network' style block that is used
        in residual connections inside a GAN

        Arguments:

        hidden_size: int
            the number of units in the network hidden layer
            processed using a convolution
        output_size: int
            the number of units in the network output layer
            processed using a convolution"""
        super(ResBlockUp, self).__init__()

        self.up = tf.keras.layers.UpSampling2D()
        self.module = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(axis=[1, 2, 3], **kwargs),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(hidden_size,
                                   3,
                                   padding='same',
                                   **kwargs),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.LayerNormalization(axis=[1, 2, 3], **kwargs),
            tf.keras.layers.ReLU(negative_slope=0.2),
            tf.keras.layers.Conv2D(output_size,
                                   3,
                                   padding='same',
                                   **kwargs)])

        # these parameters need to be stored so that
        # tf.keras.model.save_model works
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        return self.up(inputs, **kwargs) + self.module(inputs, **kwargs)

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            keras base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      output_size=self.output_size,
                      ** self.kwargs)

        base_config = super(ResBlockUp, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
