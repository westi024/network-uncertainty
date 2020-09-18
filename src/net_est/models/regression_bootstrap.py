"""

This function creates the training data and fits multiple neural network models.

This file replicates the results from
"A Study of the bootstrap method for estimating the accuracy of artificial neural networks in predicting
nuclear transient processes" https://ieeexplore.ieee.org/document/1645061

"""

import os
import numpy as np
import ray
import json
from ray import tune
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

from net_est.data.data_generator import generate_training_data, noise_function
from net_est.utils.resource_config import configure_cpu_gpu_resources, create_results_directory
from net_est.models import network_builder as net_build
from net_est.utils.timing import timer
from net_est.utils.config_loader import load_config


class bootstrap_trainer(tune.Trainable):
    """  Trains and evaluates a model using the Trainable API.

    Attributes
    ----------
    model_name: str
        The model name used for indexing the correct fold of the training data
    train_data: tf.data.Dataset

    val_data: tf.data.Dataset

    model: keras.Model
        The model for this fold of the training data
    """
    @timer
    def setup(self, config):
        """ Loads the training data and builds the model """
        # Set the model name so we can index the correct K fold
        self.model_name = config['model_name']

        # Load one fold of the data used by this model,
        self.train_data, self.val_data = self.load_data(config)

        # Build the model
        self.model = net_build.noisy_sin_network(config)

    def step(self):
        """ Performs model training """
        self.model.fit(self.train_data,
                       validation_data=self.val_data,
                       epochs=self.config.get('epochs', 10))

        _, error = self.model.evaluate(self.val_data)

        return {'mse': error}

    def load_data(self, config):
        """ Uses the model name to load a specific fold of the training data """
        x_train, y_train, x_val, y_val = [config['training_data'][self.model_name][x] for x in ['x_train',
                                                                                                'y_train',
                                                                                                'x_val',
                                                                                                'y_val']]
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        SHUFFLE_BUFFER_SIZE = 100
        BATCH_SIZE = config.get('batch_size', 64)

        # Shuffle and batch the datasets
        train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        val_data = val_data.batch(BATCH_SIZE)

        return train_data, val_data

    def save_checkpoint(self, tmp_checkpoint_dir):
        """ Saves the model during this training iteration as model.h5"""
        file_path = tmp_checkpoint_dir + "/model.h5"
        self.model.save_weights(file_path)
        return file_path

    def load_checkpoint(self, path):
        """ Loads a model using the parameters saved in params.json and the path"""
        del self.model
        with open(os.path.join(path, "params.json"), 'r') as f:
            config = json.load(f)
        self.model = net_build.noisy_sin_network(config)
        self.model.load_weights(os.path.join(path, "model.h5"))


def load_training_val_data(n_models=5, model_samples=500):
    """  Calls data_generator.py to create the training data set.  Applies k-fold partitioning so models are
    trained on unique data sets.

    Parameters
    ----------
    n_models: int
        The number of models
    model_samples: int
        The number of samples to generate per model

    Returns
    -------
    k_fold_training: dict
        Each model is specified with the sub-dictionary containing the model training and validation

    """
    if not isinstance(n_models, int):
        raise ValueError("The n_models argument must be an integer")

    x, y = generate_training_data(n_samples=model_samples*n_models)
    x, y = [g[:, np.newaxis] for g in [x, y]]

    # Create a validation set we'll use for reporting errors across models
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

    k_fold_training = {}
    kf = KFold(n_splits=n_models)
    fold_cnt = 0
    for train_ix, test_ix in kf.split(train_x):
        k_fold_training[f"Model_{fold_cnt}"] = {
            "x_train": train_x[train_ix, :],
            "y_train": train_y[train_ix, :],
            "x_val": train_x[test_ix, :],
            "y_val": train_y[test_ix, :]
        }
        fold_cnt += 1

    # Don't touch these until we're finished
    k_fold_training['x_test'] = test_x
    k_fold_training['y_test'] = test_y
    return k_fold_training


def create_ray_train_spec(cpu_per_job=1, config_name='noisy_sin', smoke_test=False):
    """  This configures the dictionary needed for tune.run() and calls ray.init().

    Parameters
    ----------
    cpu_per_job: int
        The number of cpus to use per training job
    config_name: str
        The name of the configuration to load from configs/regression_config.yml
    smoke_test: bool
        Flag indicating we're testing the code

    Returns
    -------
    train_spec: dict
        The dictionary passed into the tune.run().

    """
    # Set up Ray to train multiple models at once, varying only the random weight initialization
    # Check hardware resources and call ray.init() and build train_spec dictionary
    gpu_per_job, num_gpus, num_cpus, object_store_memory = configure_cpu_gpu_resources()

    # Start Ray
    ray.init(num_cpus=num_cpus,
             num_gpus=num_gpus,
             object_store_memory=object_store_memory,
             local_mode=smoke_test)

    model_config = load_config(config_name)

    # Build the train_spec dict
    train_spec = {"resources_per_trial": {
        "cpu": cpu_per_job,
        "gpu": gpu_per_job
    }, "stop": {
        "training_iteration": model_config.get('training_iterations', 5),
    }, "config": {**model_config}, 'checkpoint_at_end': True, 'checkpoint_freq': 1,
        'num_samples': 1}

    # Then load training data, don't need object store here, data is small
    k_fold_training = load_training_val_data(n_models=model_config.get('n_models', 3))
    train_spec['config']['training_data'] = k_fold_training
    train_spec['config']['name'] = config_name

    # This is how we'll index the training data
    model_names = [f"Model_{x}" for x in range(model_config.get('n_models', 3))]
    train_spec['config']['model_name'] = tune.grid_search(model_names)
    return train_spec


def train_bootstrap_models(smoke_test=False):
    train_obj = create_ray_train_spec(smoke_test=smoke_test)
    exp_dir, exp_name_dir = create_results_directory(train_obj['config']['name'])

    # This is where Ray.tune is called using the class that inherits from tune.Trainable
    _ = tune.run(
        bootstrap_trainer,
        local_dir=exp_dir,
        name=exp_name_dir,
        **train_obj
    )
    return os.path.join(exp_dir, exp_name_dir)


def save_model_results(model_dir):
    ensemble = EnsembleModel(model_dir=model_dir)
    X, Y = generate_training_data(n_samples=500)
    Y_hat = ensemble.predict(X)

    y_boot = np.squeeze(np.mean(np.array(Y_hat), axis=0))
    var_boot = np.squeeze(np.var(np.array(Y_hat), axis=0))

    ix = np.argsort(X)
    fig, axes = plt.subplots(2, 1, figsize=(15, 15))
    ax = axes.ravel()

    # Plot the prediction intervals out to 2 sigma
    ax[0].plot(X[ix], y_boot[ix], 'k-', lw=1.0, label=r'$\hat{y}_{boot}$')
    ax[0].fill_between(X[ix], y1=y_boot[ix], y2=y_boot[ix] + np.sqrt(var_boot[ix]), facecolor='y', label=r'$\sigma$')
    ax[0].fill_between(X[ix], y1=y_boot[ix], y2=y_boot[ix] - np.sqrt(var_boot[ix]), facecolor='y')

    ax[0].fill_between(X[ix], y1=y_boot[ix] + np.sqrt(var_boot[ix]),
                       y2=y_boot[ix] + 2*np.sqrt(var_boot[ix]), facecolor='c', label=r'$2\sigma$')
    ax[0].fill_between(X[ix], y1=y_boot[ix] - np.sqrt(var_boot[ix]),
                       y2=y_boot[ix] - 2 * np.sqrt(var_boot[ix]), facecolor='c')

    ax[0].scatter(X[ix], Y[[ix]], c='k', marker='x', s=0.1)
    ax[0].legend(fontsize=18)
    ax[1].scatter(X[ix], var_boot[ix], c='r', s=4, label=r'\hat{\sigma}^2')
    ax[1].plot(X[ix], noise_function(X[ix])[1], 'k-', lw=0.5, label=r'\sigma^2')
    print(f"Saving results to {model_dir}")
    plt.savefig(os.path.join(model_dir, 'ensemble_pred.png'))


def bootstrap_modeling(analyze_results=False, model_dir=None):
    """ Main function for fitting and analyzing bootstrap models

    Parameters
    ----------
    analyze_results: bool
        Flag indicating if we only want to analyze previously trained model results
    model_dir: path
        The path to the model directory for analyzing results.  If analyze_results=False, this isn't used

    Returns
    -------
    None

    """
    if analyze_results:
        if model_dir is None:
            raise ValueError("Must provide a model directory to analyze results!")
        else:
            # Report Ensemble Results -- Create Figure 6 from paper
            save_model_results(model_dir)
    else:
        model_dir = train_bootstrap_models(smoke_test=False)
        save_model_results(model_dir)


class EnsembleModel:
    def __init__(self,
                 model_dir):

        self.model_dir = model_dir
        self.ensemble = self.build_ensemble()

    def build_ensemble(self):
        """ Iterates through model_dir loading a list of models"""
        model_ensemble = []
        for model_folder in os.listdir(self.model_dir):
            if os.path.isdir(os.path.join(self.model_dir, model_folder)):
                model_ensemble.append(net_build.load_noisy_sin(model_directory=os.path.join(self.model_dir, model_folder),
                                                               metric='mse',
                                                               mode='min'))
        return model_ensemble

    def predict(self, x):
        """ Returns a list of model predictions for a sample, x"""
        predictions = []
        for model in self.ensemble:
            predictions.append(model.predict(x))

        return predictions


if __name__ == '__main__':
    bootstrap_modeling(analyze_results=True,
                       model_dir="/results/noisy_sin/tmppzm7llsl")