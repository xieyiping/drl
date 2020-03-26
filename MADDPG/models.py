import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp

HIDDEN_SIZES = (64, 64)
ACTIVATION = tf.nn.relu
LR_A = 0.0001
LR_C = 0.0002
INI = tf.initializers.glorot_normal
OPT = tf.optimizers.Adam

class CtsPolicy():
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    """
    def __init__(self, s_shape, a_shape, action_bound, init=INI, hidden_sizes=HIDDEN_SIZES, lr=LR_A):
        

        a_num = a_shape[0]

        input_ = keras.Input(shape=s_shape)
        x = keras.layers.Flatten()(input_)
        for units in HIDDEN_SIZES:
            x = keras.layers.Dense(
                    units=units,
                    activation=ACTIVATION,
                    kernel_initializer=init())(x)

        x = keras.layers.Dense(
                    units=a_num,
                    activation=tf.tanh,
                    kernel_initializer=init())(x)

        mean = keras.layers.Lambda(lambda x: x * action_bound)(x)
        
        self.model = keras.Model(inputs=input_, outputs=mean)
        self.model.compile(optimizer=OPT(lr), loss='mse')
        
        log_stdev = tf.Variable(
            tf.Variable(np.full((a_num,), 0.2, dtype=np.float32))) # todo
        self.model.trainable_weights.append(log_stdev)
        self.model.log_stdev = log_stdev


    def __call__(self, x):
        """
        x shape: (BATCH_SIZE, state_shape)
        output shape: (BATCH_SIZE, ACTION_SHAPE), (ACTION_SHAPE,)
        """
        
        mean = self.model(x)

        std = tf.math.exp(self.model.log_stdev)
        pi = tfp.distributions.Normal(mean, std)
        
        return pi

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def optimizer(self):
        return self.model.optimizer


class DDPGValueNet():
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    """
    def __init__(self, feature_shapes, a_shapes, init=INI, 
                 hidden_sizes=HIDDEN_SIZES, lr=LR_C, n_agent=None):
        input_s = [keras.Input(shape=obs_shape) for obs_shape in feature_shapes]
        input_a = [keras.Input(shape=a_shape) for a_shape in a_shapes]
        inputs  = input_s + input_a
        x       = keras.layers.Concatenate(axis=1)(inputs)
        for units in HIDDEN_SIZES:
            x = keras.layers.Dense(
                    units=units,
                    activation=ACTIVATION,
                    kernel_initializer=init())(x)

        value = keras.layers.Dense(units=1,
                                   kernel_initializer=init())(x)
        self.model = keras.Model(inputs=inputs, outputs=value)
        self.model.compile(optimizer=OPT(lr), loss='mse')

    def __call__(self, x):
        return self.model(x)
    
    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def optimizer(self):
        return self.model.optimizer