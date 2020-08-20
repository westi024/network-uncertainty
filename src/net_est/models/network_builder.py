"""

This file builds neural network models from a parameter dictionary.

"""
from tensorflow import keras
from tensorflow.keras import layers


def get_loss_function(params):
    if params['loss'] == 'mse':
        loss = keras.losses.mean_squared_error
    else:
        raise NotImplementedError("Only MSE implemented at this time")
    return loss


def get_optimizer(params):
    if params['optimizer'] == 'RMSProp':
        optimizer = keras.optimizers.RMSprop()
    else:
        raise NotImplementedError("Only RMSProp implemented at this time.")
    return optimizer


def noisy_sin_network(params):
    """ Builds a basic regression network

    Parameters
    ----------
    params: dict
        All necessary parameters to build the model.

    Returns
    -------
    model: Keras Model

    """

    # Initialize weight regularizer, initializer,
    input_shape = params['input_shape']
    network_input = layers.Input(shape=input_shape)

    # Build the hidden layers
    h = network_input
    for n in range(params['hidden_layers']):
        h = layers.Dense(units=params['nodes'][n],
                         activation=params['activation'],
                         use_bias=True,
                         kernel_initializer=params['kernel_init'],
                         kernel_regularizer=params['kernel_reg'])(h)

    # Build the output layer
    output_shape = params['output_shape']
    output_layer = layers.Dense(units=output_shape,
                                activation='linear',
                                use_bias=True,
                                kernel_initializer=params['kernel_init'],
                                kernel_regularizer=params['kernel_reg'])

    model = keras.Model(inputs=network_input, outputs=output_layer, name='noisy_sin_model')
    print(model.summary())
    loss_function = get_loss_function(params)
    optimizer = get_optimizer(params)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mse'])
    return model

