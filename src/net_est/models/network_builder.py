"""

This file builds neural network models from a parameter dictionary.

"""
import numpy as np
import os
import json
import glob
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mdn_activation(x):
    """Activation function used when predicting variance """
    return tf.keras.activations.elu(x) + 1


def mve_log_loss(y_true, y_pred):
    """ Implements Equation 27 from
    "Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances"

    Implementation based on:
    https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e

    Parameters
    ----------
    y_true: tensor (batch, n_dim)
        The truth value, no sigma information included.
    y_pred: tensor (batch, 2 * n_dim)
        The concatenate sigma and mean

    """
    n_dim = int(y_pred.shape[1] / 2)
    sigma = y_pred[:, 0:n_dim]
    mu = y_pred[:, n_dim:]

    log_likelihood = -0.5 * tf.reduce_sum(tf.math.log(sigma) + tf.square(y_true - mu)/sigma, axis=1)
    return tf.reduce_mean(-log_likelihood)


def get_loss_function(params):
    if params['loss'] == 'mse':
        loss = keras.losses.mean_squared_error
    elif params['loss'] == 'log-likelihood':
        loss = mve_log_loss
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


def build_noisy_sin(params, use_dropout=False, dropout_rate=0.2):
    """ Builds noisy sin model using params loaded from yaml file. """
    # Initialize weight regularizer, initializer,
    input_shape = params['input_shape']
    network_input = layers.Input(shape=input_shape, name='Input')

    # Build the hidden layers
    h = network_input
    for n in range(params['hidden_layers']):
        h = layers.Dense(units=params['nodes'][n],
                         activation=params['activation'],
                         use_bias=True,
                         kernel_initializer=params['kernel_init'],
                         kernel_regularizer=params['kernel_reg'],
                         name=f'hidden_{n}')(h)
        if use_dropout:
            h = layers.Dropout(dropout_rate, name=f"dropout_{n}")(h, training=True)

    # Build the output layer
    output_shape = params['output_shape']
    if params.get('sigma_net', False):
        activation = mdn_activation
    else:
        activation = tf.keras.activations.linear
    output_layer = layers.Dense(units=output_shape,
                                activation=activation,
                                use_bias=True,
                                kernel_initializer=params['kernel_init'],
                                kernel_regularizer=params['kernel_reg'],
                                name='output_layer')(h)
    network_name = 'noisy_sin'
    if params.get("sigma_net", False):
        if params.get('mean_net', None):
            network_name = 'noisy_sin_sigma'
            print("Configuring Sigma Network Output using previously trained mean network")
            mean_net = params['mean_net']
            mean_net.trainable = False
            mean_pred = mean_net(network_input)
            output_layer = layers.Concatenate(name='sigma_mean_concat')([output_layer, mean_pred])

    model = keras.Model(inputs=network_input,
                        outputs=output_layer,
                        name=network_name)
    print(model.summary())
    return model


def noisy_sin_network(params, use_dropout=False, dropout_rate=0.2):
    """ Builds a basic regression network

    Parameters
    ----------
    params: dict
        All necessary parameters to build the model.

    Returns
    -------
    model: Keras Model

    """
    model = build_noisy_sin(params, use_dropout, dropout_rate)
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


def noisy_sin_sigma_network(params, mean_network, use_dropout=False, dropout_rate=0.2):
    params['sigma_net'] = True
    params['mean_net'] = mean_network

    model = build_noisy_sin(params, use_dropout, dropout_rate)
    loss_function = get_loss_function(params)
    optimizer = get_optimizer(params)
    model.compile(loss=loss_function,
                  optimizer=optimizer)
    return model
