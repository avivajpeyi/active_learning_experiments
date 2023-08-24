"""
! pip install git+https://github.com/ziatdinovmax/gpax -q
"""
import gpax
import numpy as np
import matplotlib.pyplot as plt
from plotting import create_gif
import os
import shutil
from plotting import plot_model_true_acq, plot_mse, plot_all_mse
from tqdm.auto import trange

from example_datasets import Dataset1, Dataset2, Dataset3

import numpyro

gpax.utils.enable_x64()  # we can swap this out for a GPU later

OUTDIR = './gpax_figures'
SEED = 0

ACQ_FUNCS = dict(
    ue=gpax.acquisition.UE,
    ei=gpax.acquisition.EI,
    thompson=gpax.acquisition.Thompson,
    ucb=gpax.acquisition.UCB,
    qUCB=gpax.acquisition.qUCB,
)

# import orderedDict
from collections import OrderedDict


class ActiveGP:
    def __init__(self, initial_X, initial_Y, generator, noise_prior, main_acq='ucb'):
        """
        :param generator: The function that generates the next point to measure
        """
        self.initial_X = initial_X
        self.initial_Y = initial_Y
        self.new_x = np.array([])
        self.new_y = np.array([])
        self.X_pred = np.linspace(*generator.in_params_range, 100)
        self.fit_rng_key, self.predict_rng_key = gpax.utils.get_keys(SEED)
        self.noise_prior = noise_prior
        self.generator = generator
        self.acq_data = OrderedDict({k: None for k in ACQ_FUNCS.keys()})
        self.main_acq = main_acq

    @property
    def train_x(self):
        return np.concatenate([self.initial_X, self.new_x])

    @property
    def train_y(self):
        return np.concatenate([self.initial_Y, self.new_y])

    def compute_acquisition_functions(self):
        self.acq_data = {}
        kwgs = dict(
            rng_key=self.predict_rng_key, model=self.gp_model,
            X=self.X_pred, noiseless=True,
        )
        pnlty = dict(penalty='inverse_distance', recent_points=self.new_x)
        kwgs = dict(
            ucb=dict(**kwgs, **pnlty, beta=4, maximize=False),
            ei=dict(**kwgs, maximize=False),
            thompson=dict(**kwgs),
            ue=dict(**kwgs, **pnlty),
            qUCB=dict(**kwgs, maximize=False, alpha=2, beta=0.25),
        )
        self.acq_data[self.main_acq] = ACQ_FUNCS[self.main_acq](
            **kwgs[self.main_acq])
        if self.main_acq != 'qUCB':
            self.acq_data[self.main_acq].ravel()


    def train(self, verbose=False):
        self.gp_model = gpax.ExactGP(1, kernel='RBF', noise_prior_dist=self.noise_prior)
        self.gp_model.fit(
            self.fit_rng_key, self.train_x, self.train_y,
            print_summary=verbose, progress_bar=verbose
        )
        self.compute_acquisition_functions()

    def __call__(self, *args, **kwargs):
        if 'X' not in kwargs:
            kwargs['X'] = self.X_pred
        y_pred, y_sampled = self.gp_model.predict(self.predict_rng_key, kwargs['X'], noiseless=True, filter_nans=True)
        return y_pred, y_sampled

    def get_next_training_point(self):
        # if self.acq_data[self.main_acq] and no nans:
        if self.acq_data[self.main_acq] is not None:
            if np.isnan(self.acq_data[self.main_acq]).any():
                print('Warning: NaNs in acq data! Returning random point.')
            else:
                max_err_idx = np.unique(self.acq_data[self.main_acq].argmax(axis=1))
                new_x = self.X_pred[max_err_idx]
                new_y = self.generator(new_x)
                return new_x, new_y
        return self.generator.sample(1)

    def plot(self, model_col='tab:orange', acq_col='tab:purple', fname=None, title=None):
        # Plot observed points, mean prediction, and acqusition function
        y_pred, y_sampled = self(self.X_pred)
        true_y = self.generator.true(self.X_pred)
        lower_b = y_pred - np.nanstd(y_sampled, axis=(0, 1))
        upper_b = y_pred + np.nanstd(y_sampled, axis=(0, 1))
        return plot_model_true_acq(
            pred_data=(y_pred, lower_b, upper_b),
            true_data=(self.X_pred, true_y),
            train_data=(self.train_x, self.train_y),
            acq_data=self.acq_data,
            fname=fname,
            title=title,
        )

    def update(self):
        new_x, new_y = self.get_next_training_point()
        self.new_x = np.append(self.new_x, new_x)
        self.new_y = np.append(self.new_y, new_y)
        self.train()

    def mse(self):
        y_pred, y_sampled = self(self.X_pred)
        true_y = self.generator.true(self.X_pred)
        return np.mean((y_pred - true_y) ** 2)


def active_learning_for_gp(dataset, label, noise_prior, n_iterations=10, n_start=5, acq='ucb'):
    outdir = f'{OUTDIR}/{label}/{acq}'
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    inital_x, initial_y = dataset.sample(n_start, seed=SEED)
    activegp = ActiveGP(inital_x, initial_y, dataset, noise_prior, acq)
    mse = np.zeros(n_iterations)
    for i in trange(n_iterations):
        activegp.update()
        mse[i] = activegp.mse()
        activegp.plot(fname=f'{outdir}/fit_{i:002d}.png', title=f'Iteration {i}')
        plt.close()
    create_gif(f"{outdir}/fit_*.png", f"{outdir}/{label}.gif", 10)
    plot_mse(mse, f'{outdir}/mse.png')
    # save mse data as txt
    np.savetxt(f'{outdir}/mse.txt', mse)
    return activegp


noise_prior = numpyro.distributions.HalfNormal(0.5)
for i, d in enumerate([Dataset1(0.1), Dataset2(0.1), Dataset3(0.1)]):
    # for acq in ACQ_FUNCS.keys():
    for acq in ['qUCB']:
        active_learning_for_gp(d, f'dataset_{i + 1}', noise_prior, 15, 4, acq)
    plot_all_mse()
