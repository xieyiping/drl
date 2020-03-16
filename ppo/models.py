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


# if __name__ == "__main__":
#     action_dim = 2
#     batch_size = 2
#     policy = CtsPolicy(action_dim=action_dim, action_bound=2)
#     value = ValueNet()
#     for i in range(1):
#         print('*' * 80)

#         s = np.random.randn(batch_size, 2, 4)
        
#         # state value
#         v = value(s)
#         print(f'v(s): {v}')
        
#         # get pi
#         pi = policy(s)
        
#         print(f'PI=  mean: {pi.mean()}, std: {pi.stddev()}')
        
#         # get a
#         a = pi.sample()
#         print(f'action from pi: {a}')
        
#         # log_a
#         log_a = pi.log_prob(a)
#         print(f'log_a: {log_a}')

#         # entropy
#         print(f'entropy: {pi.entropy()}')
