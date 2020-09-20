"""
This file provides data generators for the data generating processes outlined in:

"A Study of the Bootstrap Method for Estimating the Accuracy of Artificial Neural Networks in Predicting Nuclear
Transient Processes"

"""
import os
import numpy as np
from scipy import stats
from scipy.integrate import quad
import matplotlib.pyplot as plt

from net_est.utils.plots import plot_sample_hist


class abs_value_dist(stats.rv_continuous):
    """
    Represents the input value distribution, Figure 1a from the paper

    """

    def _cdf(self, x):
        return np.where(x < 0.0, self.neg_cdf(x), self.pos_cdf(x))

    @staticmethod
    def neg_cdf(x_in):
        if x_in < -1.0:
            return 0.0
        else:
            return (-1*x_in**2.0 / 2.0) + (1/2)

    @staticmethod
    def pos_cdf(x_in):
        if 0.0 < x_in < 1.0:
            return ((x_in**2) / 2.0) + (1/2)
        else:
            return 1.0


def target_function(x):
    """ Target values

    Equation 16 from paper

    Parameters
    ----------
    x: 1darray
        The input x values

    Returns
    -------
    y: 1darray
        The target values
    """
    y = np.sin(np.pi * x) * np.cos((5/4) * np.pi * x)
    return y


def noise_function(x):
    """ The Gaussian white noise function

    Equation 17
    
    Parameters
    ----------
    x: 1darray
        The input x values

    Returns
    -------
    error: 1darray
        The error values to add to each truth value

    """
    x_ = np.pi * x
    sigma_e_squared = 0.0025 + (0.0025 * (1 + np.sin(x_))**2)
    return np.random.normal(loc=0.0, scale=sigma_e_squared), sigma_e_squared


def generate_training_data(n_samples=50, create_plot=False):
    """ Generates x, y pairs

    Parameters
    ----------
    n_samples: int
        The number of x,y pairs to generate

    create_plot: bool
        Flag to indicate if a 4 panel plot should be created matching Figure 1 from the paper.

    Returns
    -------
    x: 1darray (n_samples,)
    y: 1darray (n_samples, )

    """

    # This is the input value distribution, we'll sample from this
    x_dist = abs_value_dist(name='x_abs')
    
    # Update the distribution support
    x_dist.a = -1.0
    x_dist.b = 1.0

    # This can be commented out, making sure we have a valid PDF.
    print(f"Area Under PDF: {quad(x_dist.pdf, -1.0, 1.0)[0]}")
    
    # Sample from the distribution
    x_sampled = x_dist.rvs(size=n_samples)
    y = target_function(x_sampled)
    sigma_e_squared_sampled, sigma_e_squared = noise_function(x_sampled)
    y_e = y + sigma_e_squared_sampled

    # Create 3 Panel Plot
    x_grid = np.arange(-1, 1.0, 0.05)
    if create_plot:
        plot_dict = {
            'X_dist': x_sampled,
            'X': x_grid,
            'Y': target_function(x_grid),
            'Y_sampled': y_e,
            'Y_var': sigma_e_squared
        }
        plot_sample_hist(plot_dict, file_path='/images', file_name='training_data.png')

    return x_sampled, y_e


if __name__ == '__main__':
    x_data, y_data = generate_training_data(n_samples=1000, create_plot=True)
