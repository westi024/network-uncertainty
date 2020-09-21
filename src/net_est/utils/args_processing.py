
import argparse


def get_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser("Network Uncertainty Code")
    parser.add_argument('-method', '--METHOD', type=str, default='dropout')
    parser.add_argument('-config_name', '--config_name', type=str, default='noisy_sin')
    parser.add_argument('-debug', '--SMOKE_TEST', type=bool, default=False)
    args = parser.parse_args()
    return args
