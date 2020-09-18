
import yaml


def load_config(config_name, config_file="regression_config.yml"):
    try:
        with open(f"/configs/{config_file}") as f:
            regression_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"{e}")

    model_config = regression_config[config_name]

    return model_config
