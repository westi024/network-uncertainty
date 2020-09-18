"""

This code considers how dropout can be used to form prediction intervals.  Only going to build a single network, so
Ray will not be needed.

"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from net_est.utils.timing import timer
from net_est.utils.config_loader import load_config
from net_est.utils.resource_config import create_results_directory
from net_est.utils.plots import plot_loss
from net_est.models import network_builder as net_build
from net_est.data.data_generator import generate_training_data





def dropout_regression(config_name='noisy_sin'):
    """ Builds and trains a regression network with dropout

    """
    model_config = load_config(config_name=config_name)

    # Create the experiment directory
    exp_dir, exp_name_dir = create_results_directory(config_name)
    if not os.path.exists(os.path.join(exp_dir, exp_name_dir)):
        os.makedirs(os.path.join(exp_dir, exp_name_dir))

    # Build the network
    model = net_build.noisy_sin_network(model_config, use_dropout=True, dropout_rate=0.2)

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
    ax = plot_loss(history, loss_keys=['loss', 'val_loss'], file_path=save_path)

    # Generate more data and test with dropout enabled


if __name__ == '__main__':
    dropout_regression()






