"""

Implements metrics for comparing prediction intervals

"""

import numpy as np


def picp(targets, lower, upper):
    """ Prediction Interval Coverage Probability

    Equation 34 and 35 from "Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances"

    Parameters
    ----------
    targets: 1darray
        The truth values
    lower: 1darray
        The lower bound for each target value
    upper: 1darray
        The upper bound for each target value

    Returns
    -------
    picp_score: float
        The mean number of target values in the interval provided by upper and lower

    """
    if targets.ndim > 1:
        raise ValueError("Only 1D Target values implemented for picp metric")

    n_test = targets.shape[0]
    up_test = np.greater_equal(upper, targets)
    low_test = np.less_equal(lower, targets)
    c = np.logical_and(up_test, low_test) * 1
    return (1/n_test) * np.sum(c)


def mpiw(upper, lower):
    """ Mean Prediction Interval Width

    Equation 36 from "Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances"

    Parameters
    ---------
    upper: 1darray
        The upper bound for each target value
    lower: 1darray
        The lower bound for each target value

    """
    if upper.ndim > 1:
        raise ValueError("Only 1D arrays implemented for mpiw metric")
    n_test = upper.shape[0]
    return (1/n_test) * np.sum(upper - lower, axis=0)


def nmpiw(mpiw, R):
    """ Normalized Mean Prediction Interval Width

    Equation 37 from "Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances"

    Parameters
    ----------
    mpiw: 1darray
        The result of calling mpiw

    R: float
        The range of the target values

    """
    return mpiw/R


def calc_cwc(nmpiw, picp, eta=50, mu=0.9):
    """ Coverage width-based criterion

    Equations 28 and 29 from "Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances"

    Parameters
    ----------
    nmpiw: 1darray
        Normalized PI width
    picp: 1darray
        Prediction interval coverage probability
    eta: float
        No discussion in the paper how to set this, default value was listed in paper
    mu: float
        Corresponds to nominal confidence level and can be set to 1 - alpha

    Returns
    -------
    cwc:

    """
    # First Calculate Gamma (Equation 39)
    low = np.greater_equal(picp, mu)
    high = np.less(picp, mu)
    gamma = (np.logical_and(low, high)) * 1.0

    # Calculate CWC (Equation 38)
    cwc_score = nmpiw*(1 + gamma * picp * np.exp(-eta * (picp - mu)))
    return cwc_score
