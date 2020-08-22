"""

This function creates the training data and fits multiple neural network models.

"""

import numpy as np
import ray
import yaml
from sklearn.model_selection import train_test_split, KFold

from net_est.data.data_generator import generate_training_data
from net_est.utils.resource_config import configure_cpu_gpu_resources


def load_training_val_data(n_models=5):
    """  Calls data_generator.py to create the training data set.  Applies k-fold partitioning so models are
    trained on unique data sets.

    Parameters
    ----------
    n_models: int
        The number of models

    Returns
    -------
    k_fold_training: dict
        Each model is specified with the sub-dictionary containing the model training and validation

    """
    if not isinstance(n_models, int):
        raise ValueError("The n_models argument must be an integer")

    x, y = generate_training_data(n_samples=1000)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

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


# Set up Ray to train multiple models at once, varying only the random weight initialization
# Check hardware resources and call ray.init() and build train_spec dictionary
gpu_per_job, num_gpus, num_cpus, object_store_memory = configure_cpu_gpu_resources()

# This will be removed later
smoke_test = True

ray.init(num_cpus=num_cpus,
         num_gpus=num_gpus,
         object_store_memory=object_store_memory,
         plasma_directory="/tmp",
         local_mode=smoke_test)

# Get the model config
with open("/configs/regression_config.yml") as f:
    regression_config = yaml.safe_load(f)
model_config = regression_config['noisy_sin']

# Build the train_spec dict


# Then load training data, don't need object store here, data is small

if __name__ == '__main__':
    load_training_val_data()
