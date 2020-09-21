
import yaml


def load_config(config_name, config_file="regression_config.yml"):
    """ Loads the config_file and parses using config_name

    Parameters
    ----------
    config_name: str
        The name of the config to use in the config_file
    config_file: str (default=regression_config.yml
        The name of the config yaml to load in /configs

    Returns
    -------
    model_config: dict
        The model configuration

    """
    if not isinstance(config_name, str):
        raise TypeError(f"The configuration name is type {type(config_name)} and must be of type string")
    if not isinstance(config_file, str):
        raise TypeError(f"The configuration file is type {type(config_file)} and must be of type string")

    try:
        with open(f"/configs/{config_file}") as f:
            regression_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"{e}")
    try:
        model_config = regression_config[config_name]
    except KeyError:
        print(f"{config_name} was not found in {config_file}")
        model_config = None
    return model_config
