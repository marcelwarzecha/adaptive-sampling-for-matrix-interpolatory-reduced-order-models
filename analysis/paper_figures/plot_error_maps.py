"""Kelvin-cell showcase (seed 30, 300-400 Hz, k = 1) and the non-adaptive pROMs at (about) the same budget.

- fig07_kc_final     : (a) samples of the showcase run on the true objective, (b) error map of its final
                       posterior-mean pROM (N = 32) (Fig. 7)
- fig10_error_maps   : error maps of LHS with the same N (32, seed 1: error near p* at the median of the
                       10 seeds) and of the closest square FFD (6x6, N = 36) (Fig. 10)

Error map: mean relative transfer-function error over 1-500 Hz on the 51x51 test grid. All maps share one
logarithmic colour scale; the star marks the true optimum p*. The box gives the median error over the grid
points within 2.5 mm of p* (as in analysis/results_summary/summarize.py).

    python plot_error_maps.py             # writes figures/fig07_kc_final, fig10_error_maps
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from plot_showcase import KC, contour_levels, samples_on_contours
from style import GREEN, PARULA, RED, REFERENCE, bound_ticks, load, require, save, to_grid

NORM = LogNorm(1e-3, 10)
NEAR_RADIUS = 2.5e-3
LHS = 'kc_nonadaptive/kc_lhs32_seed01.mat'
FFD = 'kc_nonadaptive/kc_ffd6x6.mat'

require('kc_showcase', 'kc_nonadaptive')
tg = load(REFERENCE).testgrid
o = list(tg.objective_ids).index('f300-400_k1')
p_star = tg.S[np.argmin(tg.lnR_true[:, o])]
near = np.linalg.norm(tg.S - p_star, axis=1) <= NEAR_RADIUS
STAR = dict(marker='*', color='w', mec='k', mew=0.8, ms=11, ls='none', zorder=8)


def axes_style(ax, u1, u2):
    bound_ticks(ax, u1[0], u1[-1], axis='x')
    bound_ticks(ax, u2[0], u2[-1], axis='y')
    ax.set(xlabel='$p_1$', ylabel='$p_2$', aspect='equal')


def error_map(ax, r):
    u1, u2, Z = to_grid(r.testgrid.S, r.testgrid.meanRelErr)
    im = ax.pcolormesh(u1, u2, Z, cmap=PARULA, norm=NORM, shading='nearest', rasterized=True)
    S = np.atleast_2d(r.samples.S)
    ax.plot(S[:, 0], S[:, 1], 'x', color=RED, ms=5, mew=1, clip_on=False, zorder=7)
    ax.plot(*p_star, **STAR, clip_on=False)
    axes_style(ax, u1, u2)
    ax.text(0.03, 0.03, f'Error near $\\mathbf{{p}}^*$: {100 * np.median(r.testgrid.meanRelErr[near]):.1f} %',
            transform=ax.transAxes, fontsize=9, bbox=dict(fc='w', ec='0.5', lw=0.5, pad=2.5), zorder=9)
    return im


def colorbar(fig, im, axs):
    cb = fig.colorbar(im, ax=axs, extend='both', fraction=0.035, pad=0.03)
    cb.set_label('Mean relative error, 1-500 Hz', fontsize=12)


def kc_final():
    r = load(KC)
    u1, u2, Z = to_grid(tg.S, tg.lnR_true[:, o])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(wspace=0.3))
    handles = samples_on_contours(axs[0], r, u1, u2, Z, contour_levels(Z))
    axs[0].plot(*p_star, **STAR, clip_on=False)
    axs[0].grid(color='0.9', lw=0.5)
    axes_style(axs[0], u1, u2)
    axs[0].set_title('(a) Samples on the true objective $\\ln R$')
    im = error_map(axs[1], r)
    axs[1].set_title(f'(b) Error of the final pROM, $N = {r.final.n_total}$')
    colorbar(fig, im, axs)
    handles.append(Line2D([], [], **{**STAR, 'ms': 10}, label=r'True optimum $\mathbf{p}^*$'))
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.45, 1.0), fontsize=9)
    save(fig, 'fig07_kc_final')


def error_maps():
    runs = [(LHS, 'LHS'), (FFD, r'FFD 6$\times$6')]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(wspace=0.3))
    for ax, (rel, lab), tag in zip(axs, runs, 'ab'):
        r = load(rel)
        im = error_map(ax, r)
        ax.set_title(f'({tag}) {lab}, $N = {int(r.n)}$')
    colorbar(fig, im, axs)
    handles = [Line2D([], [], ls='none', marker='x', color=RED, mew=1, label='Samples'),
               Line2D([], [], **{**STAR, 'ms': 10}, label=r'True optimum $\mathbf{p}^*$')]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.45, 0.93), fontsize=9)
    save(fig, 'fig10_error_maps')


if __name__ == '__main__':
    kc_final()
    error_maps()
