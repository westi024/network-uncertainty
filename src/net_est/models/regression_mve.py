"""
Implements Mean-Variance Estimation as described in:
"Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances"
and "Estimating the mean and variance of the target probability distribution"
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from net_est.utils.timing import timer
from net_est.utils.config_loader import load_config
import net_est.utils.resource_config as resource_util
from net_est.data.data_generator import generate_training_data, target_function, noise_function
from net_est.models import network_builder as net_build
from net_est.utils.plots import plot_prediction_interval


@timer
def setup_datasets(config):
    """ Creates training data and splits into two datasets, D1 and D2.

    D1 is used to train the mean network, using D2 as validation and vice-versa for the sigma network.

    Parameters
    ----------
    config: dict

    Returns
    -------
    D1: Dataset
    D2: Dataset

    """
    x, y = generate_training_data(n_samples=config.get('n_samples', 1000))
    x, y = [g[:, np.newaxis] for g in [x, y]]

    # Generate training data and split into dataset 1 (D1) and dataset 2 (D2)
    D1_x, D2_x, D1_y, D2_y = train_test_split(x, y, test_size=0.20)
    D1 = tf.data.Dataset.from_tensor_slices((D1_x, D1_y))
    D2 = tf.data.Dataset.from_tensor_slices((D2_x, D2_y))

    # Shuffle and batch the datasets
    SHUFFLE_BUFFER_SIZE = 100
    BATCH_SIZE = config.get('batch_size', 64)
    D1 = D1.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    D2 = D2.batch(BATCH_SIZE)

    return D1, D2


def setup_training(config_name):
    # Load the model config
    model_config = load_config(config_name=config_name)
    model_config = model_config['mve']

    # Create the experiment directory
    exp_dir, exp_name_dir = resource_util.setup_exp_directory(config_name=config_name)

    return model_config, exp_dir, exp_name_dir


def mve_regression(config_name='noisy_sin'):
    """  First trains mean network using MSE loss then trains sigma network use log-likelihood. Both networks are
    trained at once after initial training to complete the 3 step training process.

    """
    # Model Training Setup
    model_config, exp_dir, exp_name_dir = setup_training(config_name)

    # Create the mean and sigma datasets
    D1, D2 = setup_datasets(model_config)

    # Build the Mean Network
    mean_config = model_config['mean_network']
    mean_network = net_build.noisy_sin_network(mean_config)

    # Phase I: Train the Mean Network before training the Sigma Network
    checkpoint_path = os.path.join(exp_dir, exp_name_dir, 'mean_model.h5')
    mean_network = train_net(mean_network, model_config['mean_network'],
                             checkpoint_path=checkpoint_path, train_data=D1, val_data=D2)

    # To build the sigma network, we need the predictions from the mean network to calculate loss
    sigma_config = model_config['sigma_network']
    sigma_network = net_build.noisy_sin_sigma_network(params=sigma_config,
                                                      mean_network=mean_network)

    # Phase II: Train the Sigma Network
    checkpoint_path = os.path.join(exp_dir, exp_name_dir, 'sigma_model.h5')
    sigma_network = train_net(sigma_network, model_config['sigma_network'],
                              checkpoint_path=checkpoint_path, train_data=D2, val_data=D1)

    # Phase III: Unlock the Mean Network weights and train both nets at the same time using log-likelihood loss
    mve_network = phase_three_training(sigma_network, model_config, exp_dir=exp_dir, exp_name_dir=exp_name_dir)

    # Report results, plot PI
    x_test_values = np.arange(-1.0, 1.0, 0.01)
    mu_sigma = mve_network.predict(x_test_values)
    n_dim = int(mu_sigma.shape[1] / 2)
    sigma_squared = mu_sigma[:, :n_dim]
    mu = mu_sigma[:, n_dim:]

    y_target = noise_function(x_test_values)[0] + target_function(x_test_values)

    # Need these values to plot prediction interval
    plot_dict = {
        'X': x_test_values,
        'Y': y_target,
        'y_mean': np.squeeze(mu),
        'y_std': np.squeeze(np.sqrt(sigma_squared)),
        'sigma_squared': noise_function(x_test_values)[1]
    }

    plot_prediction_interval(plot_dict, file_path=os.path.join(exp_dir, exp_name_dir), file_name='mve_interval.png')


def phase_three_training(trained_network, model_config, exp_dir, exp_name_dir):
    """ Updates both the mean and sigma network weights using negative log-likelihood loss """
    # Resample the training data
    D1, D2 = setup_datasets(model_config)

    # Make the mean network trainable
    trained_network.get_layer('noisy_sin').trainable = True

    checkpoint_path = os.path.join(exp_dir, exp_name_dir, 'mve_net.h5')
    mve_network = train_net(trained_network, model_config['sigma_network'],
                            checkpoint_path=checkpoint_path, train_data=D2, val_data=D1)
    return mve_network


def train_net(model, config, checkpoint_path, train_data, val_data):
    # Train the network
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=config.get('patience', 10),
            factor=config.get('lr_reduce_factor', 0.1),
            verbose=1,
            mode='auto',
            min_delta=1e-3,
            cooldown=config.get('cooldown', 2),
            min_lr=config.get('min_lr', 1e-10)
        )
    ]
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=config.get('epochs', 10),
                        callbacks=callbacks)
    model.load_weights(checkpoint_path)

    # May want to visualize the loss plots?

    return model


if __name__ == '__main__':
    mve_regression(config_name='noisy_sin')
