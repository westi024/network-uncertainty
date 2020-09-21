"""

Main function for running scripts


"""
import os
abs_path = os.path.abspath(__file__)
dname = os.path.dirname(abs_path).split("net_est")[0]
os.chdir(dname)

from net_est.models.regression_dropout import dropout_regression

dropout_regression()
