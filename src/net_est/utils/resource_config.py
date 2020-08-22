
import tensorflow as tf
import psutil
import os
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
