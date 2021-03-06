from wgangp.layers.res_block import ResBlock
from wgangp.layers.res_block_up import ResBlockUp
import tensorflow as tf


class Generator(tf.keras.Sequential):

    def __init__(self,
                 hidden_size,
                 output_size,
                 num_layers,
                 **kwargs):
        """Creates a 'network-in-network' style block that is used
        in residual connections inside a GAN

        Arguments:

        hidden_size: int
            the number of units in the network hidden layer
            processed using a convolution
        output_size: int
            the number of units in the network output layer
            processed using a convolution
        num_layers: int
            the number of layers of downsampling and upsampling
            each layer consists of residual connections"""

        a = tf.keras.layers.Dense(
            16 * output_size, **kwargs)

        b = tf.keras.layers.Reshape(
            [4, 4, output_size])

        layers = [a, b]

        for layer in reversed(range(num_layers)):

            a = ResBlock(hidden_size,
                         output_size,
                         **kwargs)

            b = ResBlockUp(hidden_size,
                           output_size,
                           **kwargs)

            layers.extend([a, b])

        a = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], **kwargs)

        b = tf.keras.layers.ReLU(negative_slope=0.2)

        c = tf.keras.layers.Conv2D(3, 3, padding='same', **kwargs)

        d = tf.keras.layers.Activation('tanh')

        layers.extend([a, b, c, d])

        super(Generator, self).__init__(layers)

        # these parameters need to be stored so that
        # tf.keras.model.save_model works
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.kwargs = kwargs

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
                      num_layers=self.num_layers,
                      ** self.kwargs)

        base_config = super(Generator, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
