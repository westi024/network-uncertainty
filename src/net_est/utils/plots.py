"""

Contains functions for plotting results and standardizing plots


"""
import numpy as np
import os
import matplotlib.pyplot as plt


def save_plot(fig, file_path, file_name):
    """ Saves the plot """
    fig.savefig(os.path.join(file_path, file_name), bbox_inches='tight', dpi=150)


def set_plot(ax, labelsize=14):
    """ Configures axis properties to make nice looking plots

    Parameters
    ----------
    ax: axes object

    labelsize: int
        The size of the axes labels

    Returns
    -------
    ax: axes object

    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params('both', labelsize=labelsize)
    return ax


def plot_loss(history_result, loss_keys=None, file_path=None):
    """ Makes the history plot showing train/val loss

    Parameters
    ----------
    history_result: History
        The returned history object from calling model.fit()
    loss_keys: Iterable
        The keys to index History with
    file_path: path

    Returns
    -------
    ax: axes object

    """

    if loss_keys is None:
        loss_keys = ['loss', 'val_loss']
    if file_path is None:
        raise ValueError("Must provide file path for saving the loss plot")

    fs = 14
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    labels = {
        'loss': 'Training',
        'val_loss': 'Validation'
    }

    for k in loss_keys:
        ax.plot(history_result.epoch, history_result.history[k], lw=0.5, label=labels.get(k, k))

    ax.set_xlabel('Epochs', fontsize=fs)
    ax = set_plot(ax, labelsize=fs)
    plt.legend(fontsize=fs)
    save_plot(fig, file_path=file_path, file_name='loss_plot.png')

    return ax


def plot_prediction_interval(data, file_path, file_name):
    """

    Parameters
    ----------
    data: dict
        Contains X, Y, y_mean, and y_std
    file_path: path
    file_name: str
        The name used to save the plot

    """

    X, Y, y_mean, y_std, sigma_squared = [data[x] for x in ['X',
                                                            'Y',
                                                            'y_mean',
                                                            'y_std',
                                                            'sigma_squared']]

    ix = np.argsort(X)
    fig, axes = plt.subplots(2, 1, figsize=(15, 15))
    ax = axes.ravel()

    ax[0].plot(X[ix], y_mean[ix], 'k-', lw=1.0, label=r'$\hat{y}_{boot}$')

    # 1 Sigma Interval
    ax[0].fill_between(X[ix], y1=y_mean[ix],
                       y2=y_mean[ix] + y_std[ix],
                       facecolor='y',
                       label=r'$\sigma$')

    ax[0].fill_between(X[ix], y1=y_mean[ix],
                       y2=y_mean[ix] - y_std[ix],
                       facecolor='y')

    # 2 Sigma Interval
    ax[0].fill_between(X[ix], y1=y_mean[ix] + y_std[ix],
                       y2=y_mean + 2 * y_std[ix],
                       facecolor='c',
                       label=r'$2\sigma$')

    ax[0].fill_between(X[ix], y1=y_mean[ix] - y_std[ix],
                       y2=y_mean - 2 * y_std[ix],
                       facecolor='c')

    ax[0].scatter(X[ix], Y[ix], c='k', marker='x', s=0.1)
    ax[0].legend(fontsize=18)

    # Plot the variance
    ax[1].scatter(X[ix], y_std[ix]**2, c='r', s=4, label=r'$\hat{\sigma}}^2$')
    ax[1].plot(X[ix], sigma_squared[ix], 'k-', lw=0.5, label=r'$\sigma$')

    # Save the plot
    save_plot(fig, file_path, file_name)













