"""

Main function for running scripts

"""
# Set the working directory
import os
abs_path = os.path.abspath(__file__)
dname = os.path.dirname(abs_path).split("net_est")[0]
os.chdir(dname)

from net_est.models.regression_dropout import dropout_regression
from net_est.models.regression_bootstrap import bootstrap_modeling
from net_est.models.regression_mve import mve_regression
from net_est.utils.args_processing import get_args

args = get_args()
implemented_methods = {
    'bootstrap': bootstrap_modeling,
    'dropout': dropout_regression,
    'mve': mve_regression
}

if args.METHOD == 'dropout':
    dropout_regression(args.config_name)
elif args.METHOD == 'bootstrap':
    bootstrap_modeling(config_name=args.config_name, smoke_test=args.SMOKE_TEST)
else:
    print("Still working on other methods for main.py")
