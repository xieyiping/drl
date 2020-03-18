import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp

HIDDEN_SIZES = (32, 32)
ACTIVATION = tf.nn.relu
LR_A = 0.0001
LR_C = 0.0002
INI = tf.initializers.glorot_normal
OPT = tf.optimizers.Adam

class CtsPolicy(keras.Model):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    """
    def __init__(self, action_dim, action_bound, init=INI, hidden_sizes=HIDDEN_SIZES, lr=LR_A):

        super(CtsPolicy, self).__init__()

        self.hidden_layers = []
        self.hidden_layers.append(keras.layers.Flatten())
        for units in HIDDEN_SIZES:
            self.hidden_layers.append(keras.layers.Dense(
                    units=units,
                    activation=ACTIVATION,
                    kernel_initializer=init()))
        self.hidden_layers.append(keras.layers.Dense(
                    units=action_dim,
                    activation=tf.tanh,
                    kernel_initializer=init()))
        self.mean = keras.layers.Lambda(lambda x: x * action_bound)
        self.log_stdev = tf.Variable(tf.zeros(action_dim))

        self.compile(optimizer=OPT(lr), loss='mse')

    def call(self, x):
        """
        x shape: (BATCH_SIZE, state_shape)
        output shape: (BATCH_SIZE, ACTION_SHAPE), (ACTION_SHAPE,)
        """

        for layer in self.hidden_layers:
            x = layer(x)
        
        mean = self.mean(x)

        std = tf.math.exp(self.log_stdev)
        pi = tfp.distributions.Normal(mean, std)
        
        return pi


class ValueNet(keras.Model):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    """
    def __init__(self, init=INI, hidden_sizes=HIDDEN_SIZES, lr=LR_C):

        super(ValueNet, self).__init__()

        self.hidden_layers = []
        self.hidden_layers.append(keras.layers.Flatten())
        for units in HIDDEN_SIZES:
            self.hidden_layers.append(keras.layers.Dense(
                    units=units,
                    activation=ACTIVATION,
                    kernel_initializer=init()))

        self.value = keras.layers.Dense(units=1,
                                        kernel_initializer=init())

        self.compile(optimizer=OPT(lr), loss='mse')

    def call(self, x):
        '''
        An example value network, with support for arbitrarily many
        fully connected hidden layers (by default 2 * 128-neuron layers),
        maps a state of size (state_dim) -> a scalar value.
        '''

        for layer in self.hidden_layers:
            x = layer(x)
        
        v = self.value(x)

        return v


class DDPGValueNet():
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    """
    def __init__(self, feature_shape, a_num, init=INI, 
                 hidden_sizes=HIDDEN_SIZES, lr=LR_C):
        
        input_s = keras.Input(shape=feature_shape)
        input_a = keras.Input(shape=(a_num,))
        x       = keras.layers.Concatenate(axis=1)([input_s, input_a])
        for units in HIDDEN_SIZES:
            x = keras.layers.Dense(
                    units=units,
                    activation=ACTIVATION,
                    kernel_initializer=init())(x)

        value = keras.layers.Dense(units=1,
                                   kernel_initializer=init())(x)
        self.model = keras.Model(inputs=[input_s, input_a], outputs=value)
        self.model.compile(optimizer=OPT(lr), loss='mse')

    def __call__(self, x):
        return self.model(x)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def optimizer(self):
        return self.model.optimizer

