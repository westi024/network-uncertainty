"""

This file builds neural network models from a parameter dictionary.

"""
import numpy as np
import os
import json
import glob
import re

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


def get_best_checkpoint_dir(model_directory, metric='mse', mode='min'):

    # Figure out which checkpoint has the best model and load the weights from that checkpoint
    checkpoint_dir = []
    for name in glob.glob(os.path.join(model_directory, "checkpoint_*")):
        checkpoint_dir.append(os.path.split(name)[-1])

    with open(os.path.join(model_directory, "result.json"), 'r') as f:
        result_dict = [json.loads(line) for line in f]

    # Sort the names https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    checkpoint_dir.sort(key=natural_keys)
    metric_results = [result_dict[i][metric] for i in range(len(result_dict))]
    check_vals = [int(x.split("_")[-1]) for x in checkpoint_dir]
    metric_results = [metric_results[x - 1] for x in check_vals]

    if mode == 'min':
        best_checkpoint_ix = np.argmin(metric_results)
    else:
        best_checkpoint_ix = np.argmax(metric_results)

    return checkpoint_dir[best_checkpoint_ix]


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
                                kernel_regularizer=params['kernel_reg'])(h)

    model = keras.Model(inputs=network_input, outputs=output_layer, name='noisy_sin_model')
    print(model.summary())
    loss_function = get_loss_function(params)
    optimizer = get_optimizer(params)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mse'])
    return model


def load_noisy_sin(model_directory, metric='mse', mode='min'):
    """

    Parameters
    ----------
    model_directory: path
        The path to the directory containing all the checkpoint folders
    metric: str
        The name of the metric we'll sort on
    mode: str
        Either max or min

    Returns
    -------
    model: Model

    """

    with open(os.path.join(model_directory, 'params.json'), 'r') as f:
        config = json.load(f)

    # Build the Model
    model = noisy_sin_network(config)
    best_model_dir = get_best_checkpoint_dir(model_directory=model_directory, metric=metric, mode=mode)
    model.load_weights(os.path.join(model_directory, best_model_dir, 'model.h5'))

    return model
