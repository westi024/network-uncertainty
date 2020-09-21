"""

This code considers how dropout can be used to form prediction intervals.  Only going to build a single network, so
Ray will not be needed.

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

from net_est.utils.timing import timer
from net_est.utils.config_loader import load_config
from net_est.utils.resource_config import create_results_directory
from net_est.utils.plots import plot_loss, plot_prediction_interval
from net_est.models import network_builder as net_build
from net_est.data.data_generator import generate_training_data, target_function, noise_function


def predict_with_dropout(model):
    """ Creates Keras backend function that enables the learning phase for making model predictions

    Based on implementation at:
    https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf

    Parameters
    ----------
    model: Model

    Returns
    -------
    predict_function: Keras Function
        Predicts model output with dropout enabled.
    """

    predict_function = K.function(model.inputs[K.learning_phase()],
                                  model.outputs)
    return predict_function


@timer
def dropout_regression(config_name='noisy_sin'):
    """ Builds and trains a regression network with dropout

    Parameters
    ----------
    config_name: str
        Name in the configs/regression_config.yml

    """
    model_config = load_config(config_name=config_name)

    # Create the experiment directory
    exp_dir, exp_name_dir = create_results_directory(config_name)
    if not os.path.exists(os.path.join(exp_dir, exp_name_dir)):
        os.makedirs(os.path.join(exp_dir, exp_name_dir))

    # Build the network
    model = net_build.noisy_sin_network(model_config, use_dropout=True, dropout_rate=0.3)

    # Load the training/validation data
    x, y = generate_training_data(n_samples=1000)
    x, y = [g[:, np.newaxis] for g in [x, y]]

    # Create a validation set we'll use for reporting errors across models
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.20)
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))

    # Shuffle and batch the datasets
    SHUFFLE_BUFFER_SIZE = 100
    BATCH_SIZE = model_config.get('batch_size', 64)
    train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_data = val_data.batch(BATCH_SIZE)

    # Train the network
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=model_config.get('epochs', 10))

    # Report training results
    save_path = os.path.join(exp_dir, exp_name_dir)
    _ = plot_loss(history, loss_keys=['loss', 'val_loss'], file_path=save_path)
    save_prediction_interval(model, file_path=save_path, file_name='dropout_interval.png')


def save_prediction_interval(model, file_path, file_name, n_iters=1000):
    """ Makes predictions with dropout enabled and saves the result

    Parameters
    ----------
    model: keras Model
        The trained neural network
    file_path: path
    file_name: str
    n_iters: int

    Returns
    -------
    None

    """
    pred_func = predict_with_dropout(model)
    y_preds = []
    x_test_values = np.arange(-1.0, 1.1, 0.01)
    for _ in range(n_iters):
        y_preds.append(pred_func(x_test_values)[0])

    y_target = noise_function(x_test_values)[0] + target_function(x_test_values)

    # Need these values to plot prediction interval
    plot_dict = {
        'X': x_test_values,
        'Y': y_target,
        'y_mean': np.squeeze(np.mean(y_preds, axis=0)),
        'y_std': np.squeeze(np.std(y_preds, axis=0)),
        'sigma_squared': noise_function(x_test_values)[1]
    }

    plot_prediction_interval(plot_dict, file_path=file_path, file_name=file_name)


if __name__ == '__main__':
    dropout_regression()
