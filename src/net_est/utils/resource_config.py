
import tensorflow as tf
import psutil
import os
import tempfile
from tensorflow.python.client import device_lib


def configure_cpu_gpu_resources():
    """  Determines number of CPUs and GPUs available on the system

    Returns
    -------
    gpu_resources: float
        The percentage of each GPU to use for each worker.
    num_gpus: int
    num_cpus: int
    object: int
        The available system memory

    """
    gpu_avail = tf.test.is_gpu_available()
    memory_available = psutil.virtual_memory()[1]
    object_store_memory = 50E9
    if memory_available < object_store_memory:
        object_store_memory = memory_available * 0.90
    num_cpus = os.cpu_count()
    num_gpus = 0
    gpu_per_job = 0.0
    if gpu_avail:
        local_device_protos = device_lib.list_local_devices()
        gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
        print(f"\n [*] This machine has {len(gpu_names)} GPUs available")
        num_gpus = len(gpu_names)
        gpu_per_job = 0.5

    return gpu_per_job, num_gpus, num_cpus, object_store_memory


def create_results_directory(config_name):
    """ Creates the results directory and a random experiment name.

    This function creates the folder structure necessary for saving Ray results without
    using the default ray_results directory.

    Parameters
    ----------
    config_name: str
        The name of the configuration to use for creating an experiment path

    Returns
    -------
    config_dir: Path
        The path within the /results directory that relies on the train_spec['config']['name']
    exp_name: str
        Random string

    """
    if not isinstance(config_name, str):
        raise TypeError(f"configuration name is of type {type(config_name)}, expected string")

    config_dir = os.path.join(f"/results/{config_name}")

    # Need a random experiment name
    random_path = tempfile.NamedTemporaryFile()
    exp_name = os.path.basename(random_path.name)

    if os.path.exists(config_dir):
        pass
    else:
        os.makedirs(config_dir)

    return config_dir, exp_name


def setup_exp_directory(config_name):
    exp_dir, exp_name_dir = create_results_directory(config_name)
    if not os.path.exists(os.path.join(exp_dir, exp_name_dir)):
        os.makedirs(os.path.join(exp_dir, exp_name_dir))

    return exp_dir, exp_name_dir
