"""Paper figures from the showcase runs (seed 30) and the full-order reference data.

- fig04_beam_iterations : surrogate surfaces of the beam showcase in selected iterations (Fig. 4)
- fig05_beam_final      : final samples of the beam showcase on the true objective (Fig. 5)
- fig12_kc_iterations   : surrogate surfaces of the Kelvin-cell showcase (Fig. 12, Appendix B)
- fig08_tf_optima       : transfer functions at the true optima for 200-400 Hz, k = 1 and k = 20 (Fig. 8)

Contour levels as in MATLAB's contour(Z, 20): 20 levels evenly between the minimum and the maximum of ln R.

Surfaces: ln R of the Thompson draw that fmincon minimizes ('TS objective', parula) and of the
posterior mean ('Objective', orange), logged on a 20x20 grid in every iteration. fmincon's path
is not logged; the panels show its random start p0 and the found optimum p_hat.

    python plot_showcase.py               # writes figures/*.{pdf,png}
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import RegularGridInterpolator

from style import (ASTERISK, BLUE, GREEN, MATLAB_BLUE, ORANGE, PARULA, RED, REFERENCE, ROOT, bound_ticks,
                   load, require, save, to_grid)

BEAM = 'beam_showcase/beam_f50-100_k1_seed30.mat'
KC = 'kc_showcase/kc_f300-400_k1_seed30.mat'
BEAM_ITERATIONS = (1, 8, 11)               # last entry: final iteration (see iterations_figure)
KC_ITERATIONS = (1, 6, 10, 15, 20, 24)
BEAM_VIEW = (30, 45)                      # (elevation, azimuth): corner (0.05, 0.05) in front
KC_VIEW = (35, 135)                       # corner (0.055, 0.045) in front


def contour_levels(Z):
    """MATLAB's contour(Z, 20): 20 levels evenly between the minimum and the maximum."""
    return np.linspace(np.nanmin(Z), np.nanmax(Z), 22)[1:-1]


def surface_legend(fig, y=1.0, ncol=4):
    handles = [Line2D([], [], color=ORANGE, lw=1.2, label='Objective (posterior mean)'),
               Patch(facecolor=PARULA(0.6), edgecolor='k', label='TS objective'),
               Line2D([], [], ls='none', marker='D', mfc='none', mec='k', ms=6, label=r'Start $\mathbf{p}_0$'),
               Line2D([], [], ls='none', marker='D', mfc=GREEN, mec='k', ms=7, label=r'Found optimum $\hat{\mathbf{p}}$')]
    fig.legend(handles=handles, loc='upper center', ncol=ncol, frameon=True, bbox_to_anchor=(0.5, y), fontsize=10)


def iteration_panel(ax, it, bounds, view, zlabel, zside):
    u1, u2, Zts = to_grid(it.surface_points, it.surface_lnR_ts)
    _, _, Zm = to_grid(it.surface_points, it.surface_lnR_mean)
    P1, P2 = np.meshgrid(u1, u2)
    lo, hi = np.nanmin(Zts), np.nanmax(Zts)
    colors = PARULA((Zts - lo) / (hi - lo))
    ax.plot_surface(P1, P2, Zts, facecolors=colors, edgecolor='k', linewidth=0.15, shade=False)
    ax.plot_wireframe(P1, P2, Zm, color=ORANGE, linewidth=0.6)
    ts = RegularGridInterpolator((u2, u1), Zts)
    z0 = np.clip(it.z0, bounds[0], bounds[1])
    ax.plot(*z0, ts(z0[::-1])[0], ls='none', marker='D', mfc='none', mec='k', ms=6, zorder=10)
    ax.plot(*it.p_hat, it.lnR_ts, ls='none', marker='D', mfc=GREEN, mec='k', ms=7, zorder=10)
    ax.view_init(*view)
    ax.zaxis.set_ticks_position(zside)          # z axis on the left, as in the paper
    ax.zaxis.set_label_position(zside)
    ax.set_xlabel('$p_1$', labelpad=8)
    ax.set_ylabel('$p_2$', labelpad=8)
    if zlabel:
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$\ln R(\mathbf{p})$', labelpad=6, rotation=90)
    for setter, k in ((ax.set_xticks, 0), (ax.set_yticks, 1)):
        t = np.linspace(bounds[0][k], bounds[1][k], 5)
        setter(t, [f'{np.round(v + 1e-12, 4):g}' for v in t])
    ax.zaxis.set_major_locator(MaxNLocator(4))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):     # white panes, light MATLAB-like grid
        axis.set_pane_color((1, 1, 1, 1))
        axis._axinfo['grid'].update(color=(0.85, 0.85, 0.85), linewidth=0.5)
    ax.tick_params(labelsize=8, pad=0)
    ax.set_title(f'Iteration {int(it.iteration)} ($N = {int(it.n_samples)}$, degree {int(it.poly_order)})', pad=0)


def iterations_figure(rel, iterations, ncols, view, name, zside='default'):
    r = load(rel)
    its = np.atleast_1d(r.iterations)
    # The last panel always shows the final iteration: a rerun on another machine can end after a different
    # number of iterations (round-off), and earlier panels beyond its end are dropped.
    iterations = [k for k in iterations[:-1] if k < len(its)] + [len(its)]
    S = r.samples.S
    bounds = (S.min(axis=0), S.max(axis=0))
    nrows = int(np.ceil(len(iterations) / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, 3.9 * nrows))
    for j, k in enumerate(iterations):
        iteration_panel(fig.add_subplot(nrows, ncols, j + 1, projection='3d'), its[k - 1], bounds, view,
                        zlabel=j % ncols == 0, zside=zside)
    fig.subplots_adjust(wspace=0.05, hspace=0.15, top=1 - 0.6 / (3.9 * nrows))
    surface_legend(fig, y=1.0)
    save(fig, name)


def samples_on_contours(ax, r, u1, u2, Z, levels):
    ax.contour(u1, u2, Z, levels=levels, colors='0.45', linewidths=0.8, linestyles='solid')
    S, origin = r.samples.S, np.asarray(r.samples.origin)
    p_hat = np.array([it.p_hat for it in np.atleast_1d(r.iterations)])
    ax.plot(S[origin == 0, 0], S[origin == 0, 1], 'o', color=RED, ms=6, clip_on=False, zorder=5)
    ax.plot(S[origin > 0, 0], S[origin > 0, 1], ls='none', marker=ASTERISK, color=BLUE, ms=7, mew=0.9,
            clip_on=False, zorder=6)
    ax.plot(p_hat[:, 0], p_hat[:, 1], 'D', color=GREEN, mec=GREEN, ms=4, clip_on=False, zorder=7)
    handles = [Line2D([], [], ls='none', marker='o', color=RED, label='Initial samples'),
               Line2D([], [], ls='none', marker=ASTERISK, color=BLUE, label='Added samples'),
               Line2D([], [], ls='none', marker='D', color=GREEN, ms=4, label='Found optima')]
    return handles


def beam_reference():
    b = sio.loadmat(glob.glob(str(ROOT / 'results/beam_reference/*.mat'))[0],
                    squeeze_me=True, struct_as_record=False)['res']
    u1, u2, Z = to_grid(b.Z, b.lnR)
    return u1, u2, Z, np.ravel(b.p_star)


def beam_final():
    r = load(BEAM)
    u1, u2, Z, _ = beam_reference()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    handles = samples_on_contours(ax, r, u1, u2, Z, contour_levels(Z))
    bound_ticks(ax, u1[0], u1[-1], axis='x')
    bound_ticks(ax, u2[0], u2[-1], axis='y')
    ax.set(xlabel='$p_1$', ylabel='$p_2$', aspect='equal')
    ax.grid(color='0.9', lw=0.5)
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.55, 1.0), fontsize=9)
    save(fig, 'fig05_beam_final')


def kc_reference():
    tg = load(REFERENCE).testgrid
    return tg, list(tg.objective_ids)


def true_contour(ax, u1, u2, Z, p_star, label_star=True):
    ax.contour(u1, u2, Z, levels=contour_levels(Z),
               colors='k', linewidths=0.9, linestyles='solid')
    ax.plot(*p_star, 'D', color=GREEN, mec=GREEN, ms=5, label=r'Optimum $\mathbf{p}^*$' if label_star else None)
    bound_ticks(ax, u1[0], u1[-1], axis='x')
    bound_ticks(ax, u2[0], u2[-1], axis='y')
    ax.set(xlabel='$p_1$', ylabel='$p_2$', aspect='equal')


def tf_optima():
    T = sio.loadmat(ROOT / 'data/FFD_51x51_1000Hz.mat', variable_names=['S', 'omegaHz', 'y'], squeeze_me=True)
    S, f, y = T['S'], np.ravel(T['omegaHz']), T['y']
    S = S if S.shape[0] > S.shape[1] else S.T
    tg, ids = kc_reference()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    band = (200, 400)
    sel = (f >= 195) & (f <= 405)
    ax.axvspan(*band, color=(0.94, 0.99, 0.94))
    for oid, k, c in (('f200-400_k1', 1, MATLAB_BLUE), ('f200-400_k20', 20, ORANGE)):
        o = ids.index(oid)
        ps = tg.S[np.argmin(tg.lnR_true[:, o])]
        j = np.argmin(np.linalg.norm(S - ps, axis=1))
        ax.semilogy(f[sel], np.abs(y[sel, j]), color=c, lw=1.4,
                    label=f'Optimum for $k = {k}$: $\\mathbf{{p}}^* = ({1e3 * ps[0]:.1f}, {1e3 * ps[1]:.1f})$ mm')
    for b, lab in zip(band, ('Lower bound', 'Upper bound')):
        ax.axvline(b, color='0.3', ls='--', lw=0.9)
    ax.set(xlim=(195, 405), xlabel='Frequency $f$ [Hz]', ylabel='Output $|y(f)|$')
    ax.legend(loc='lower left', fontsize=10)
    save(fig, 'fig08_tf_optima')


if __name__ == '__main__':
    require('beam_reference', 'beam_showcase', 'kc_showcase')
    iterations_figure(BEAM, BEAM_ITERATIONS, 3, BEAM_VIEW, 'fig04_beam_iterations')
    beam_final()
    iterations_figure(KC, KC_ITERATIONS, 3, KC_VIEW, 'fig12_kc_iterations', zside='upper')
    tf_optima()
