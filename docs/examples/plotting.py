from matplotlib import rcParams
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import colorsys
from matplotlib.colors import ColorConverter

import glob
import numpy as np
import re
from typing import Dict, Tuple

import imageio.v3 as iio

try:
    from pygifsicle import optimize
except ImportError:
    def optimize(*args, **kwargs):
        pass

rcParams.update({"xtick.major.pad": "7.0"})
rcParams.update({"xtick.major.size": "7.5"})
rcParams.update({"xtick.major.width": "1.5"})
rcParams.update({"xtick.minor.pad": "7.0"})
rcParams.update({"xtick.minor.size": "3.5"})
rcParams.update({"xtick.minor.width": "1.0"})
rcParams.update({"ytick.major.pad": "7.0"})
rcParams.update({"ytick.major.size": "7.5"})
rcParams.update({"ytick.major.width": "1.5"})
rcParams.update({"ytick.minor.pad": "7.0"})
rcParams.update({"ytick.minor.size": "3.5"})
rcParams.update({"ytick.minor.width": "1.0"})
rcParams.update({"font.size": 20})


ACQ_COLORS = dict(
    ucb="tab:purple",
    ei="tab:green",
    thompson="tab:red",
    ue="tab:blue"
)

def remove_spines(ax):
    for spine in ["top", "right", 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def create_gif(image_regex, gif_path, duration=1, bounce=True):
    image_filepaths = glob.glob(image_regex)
    image_filepaths = sorted(image_filepaths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images = [iio.imread(filepath) for filepath in image_filepaths]
    if bounce: # images backwards skipping first and last
        images = images + images[-2:0:-1]
    iio.imwrite(gif_path, images, duration=duration, loop=0)
    optimize(gif_path)


def plot_model_true_acq(
        pred_data: Tuple[np.array, np.array, np.array],
        train_data: Tuple[np.array, np.array],
        true_data: Tuple[np.array, np.array],
        acq_data: Dict[str, np.array],
        fname='',
        title='',
        model_col="tab:orange",
        acq_col="tab:purple",
):
    """Plot the model, true function, and acquisition function."""
    ypred_med, ypred_low, ypred_up = pred_data
    xtrue, ytrue = true_data
    xtrain, ytrain = train_data
    xlims = (min(xtrue), max(xtrue))
    ylims = (min(ytrue), max(ytrue))
    # extend by ylim 10% on both sides
    yextend = np.abs(ylims[1] - ylims[0]) * 0.1
    ylims = (ylims[0] - yextend, ylims[1] + yextend)

    fig, ax = plt.subplots(1, 1, dpi=100, figsize=(10, 5))


    # PLOT MODEL
    ax.plot(xtrue, ytrue, lw=3, ls='--', c='k', label='True', alpha=0.1, zorder=-10)
    ax.scatter(xtrain, ytrain, c='k', label="Observations", zorder=10)
    ax.plot(xtrue, ypred_med, lw=2, c=model_col, label='Model', zorder=100)
    ax.fill_between(xtrue, ypred_low, ypred_up, color=model_col, alpha=0.3, zorder=100)


    # PLOT ACQUISITION FUNCTIONS
    num_acq = len(acq_data)
    ax2 = ax.twinx()
    styles = ["-", "--", "-.", ":"]
    for i, (key, acq) in enumerate(acq_data.items()):
        if key != 'qUCB':
            _plot_acq(acq, ax, ax2, xtrue, key, i, styles)
        else:
            for j, acq_i in enumerate(acq):
                _plot_acq(acq_i, ax, ax2, xtrue, f"{i} {key}", i, styles)


    # format axes
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax2.set_ylim([-1,1])
    ax2.set_xlim(xlims)
    remove_spines(ax)
    remove_spines(ax2)

    ax.set_zorder(ax2.get_zorder()+1)
    ax2.set_zorder(ax.get_zorder()-10)
    ax.patch.set_visible(False)
    ax2.patch.set_visible(False)


    if title:
        ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=12, verticalalignment="top", )

    if fname:
        plt.tight_layout()
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def _plot_acq(acq, ax, ax2, xtrue, key, i, styles):
    acq_pt = np.argmax(acq)
    acq = scale_data(acq, [-0.75, 0.75])
    label = key + " ☑" if i == 0 else key
    s = 90 if i == 0 else 30
    lw = 2.5 if i == 0 else 1.5
    ls = styles[i % len(styles)]
    clr = ACQ_COLORS.get(key, "tab:blue")
    alpha = 0.9 if i == 0 else 0.45
    ax2.plot(xtrue, acq, lw=lw, color=clr, alpha=alpha, zorder=-100, ls=ls)
    ax.plot([], [], lw=lw, color=clr, alpha=alpha, zorder=-100, ls=ls, label=label)
    ax2.scatter(xtrue[acq_pt], acq[acq_pt], s=s, color=clr, zorder=-100)

def scale_color_brightness(rgb=None, color_str="", scale_l=1.0):
    if color_str != "":
        rgb = ColorConverter.to_rgb(color_str)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def scale_data(d, new_lim):
    """Scale d to be between new_lim"""
    a, b = np.min(new_lim), np.max(new_lim)
    dmin, dmax = np.min(d), np.max(d)
    newd = (b - a) * (d - dmin) / (dmax - dmin) + a
    return newd




def plot_mse(mse, fname):
    plt.plot(mse)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    # show 0 line
    plt.axhline(0, c='k', lw=1, ls='--')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()




def test_plot():
    from example_datasets import Dataset1
    from scipy.interpolate import interp1d
    data1 = Dataset1(noise_sigma=0.1)

    samples = data1.sample(5, seed=0)
    xtrain, ytrain = samples[0], samples[1]

    xtrue = np.linspace(*data1.in_params_range, 100)
    ytrue = data1.true(xtrue)
    idx = np.argsort(xtrue)
    xtrue, ytrue = xtrue[idx], ytrue[idx]

    ypreds = []
    for i in range(50):
        samples = data1.sample(30, seed=i)
        xtrain, ytrain = samples[0], samples[1]
        f = interp1d(xtrain, ytrain, kind='cubic', fill_value='extrapolate')
        ypreds.append(f(xtrue))
    ypreds = np.array(ypreds)
    ypred_med = np.median(ypreds, axis=0)
    ypred_low = np.quantile(ypreds, 0.05, axis=0)
    ypred_up = np.quantile(ypreds, 0.95, axis=0)

    mse = (ytrue - ypred_med) ** 2

    plot_model_true_acq(
        pred_data=(ypred_med, ypred_low, ypred_up),
        train_data=(xtrain, ytrain),
        true_data=(xtrue, ytrue),
        acq_data=dict(mse=mse),
        fname='test.png',
    )


def plot_all_mse():
    FN = "./gpax_figures/dataset_{i}/{acq}/mse.txt"

    for i in [1, 2, 3]:
        plt.figure()
        for acq, col in ACQ_COLORS.items():
            mse = np.loadtxt(FN.format(i=i, acq=acq))
            plt.plot(mse, color=col, label=acq)
        plt.legend()
        plt.axhline(0, color='k', linestyle='--')
        plt.ylabel("MSE")
        plt.xlabel("Iteration")
        plt.savefig(f'./gpax_figures/dataset_{i}/mse.png')
        plt.tight_layout()
        plt.close()


if __name__ == '__main__':
    test_plot()