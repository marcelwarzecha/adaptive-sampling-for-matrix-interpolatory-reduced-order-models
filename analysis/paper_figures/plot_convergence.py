"""Convergence of the adaptive Kelvin-cell runs: exploration samples, gap to the true optimum and accuracy.

fig09_convergence (Fig. 9), from the 40 adaptive Kelvin-cell runs (4 objectives x 10 seeds):
  (a) per iteration: share of the added samples that explore (more than 2.5 mm from p*, not moved by
      the distance check), that are near-optimal (within 2.5 mm), and that the distance check moved; a moved
      candidate coincides with an existing sample, i.e., the optimizer found the same optimum again (refinement)
  (b) per iteration: true gap ln R(p_hat) - ln R* of the found optimum and the prediction error
      |ln R_mean(p_hat) - ln R(p_hat)| of the posterior-mean pROM there (median and interquartile range)
  (c) per objective: mean relative error over 1-500 Hz near p* (median over the grid points within 2.5 mm)
      vs. the number of samples N. Adaptive: replay of the posterior-mean pROM after every iteration
      (results/kc_replay; median and IQR over the runs that reached N, where at least 5 of the 10 runs did). Non-adaptive: LHS (median and range
      over the seeds) and square FFD grids at their N (results/kc_nonadaptive).

The true ln R at p_hat is interpolated (linearly) from the 51x51 reference grid (0.5 mm spacing), except
where the sample added at p_hat itself was evaluated with the FOM.

    python plot_convergence.py            # writes figures/fig09_convergence and fig13_true_objectives
"""
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from style import BLUE, MATLAB_BLUE, ORANGE, RED, REFERENCE, RES, load, require, save, to_grid

NEAR_RADIUS = 2.5e-3
GAP_NEAR = 0.02                           # appendix figure: near-optimal region of the true objective
MIN_RUNS = 5                              # (c): plot the adaptive median where at least half of the runs are running
TITLES = {'f300-400_k1': '300-400 Hz, $k = 1$', 'f300-400_k20': '300-400 Hz, $k = 20$',
          'f200-400_k1': '200-400 Hz, $k = 1$', 'f200-400_k20': '200-400 Hz, $k = 20$'}

require('kc_showcase', 'kc_adaptive', 'kc_nonadaptive', 'kc_replay', 'beam_reference', 'beam_adaptive')
tg = load(REFERENCE).testgrid
ids = list(tg.objective_ids)
L_star = {oid: tg.lnR_true[:, o].min() for o, oid in enumerate(ids)}
p_star = {oid: tg.S[np.argmin(tg.lnR_true[:, o])] for o, oid in enumerate(ids)}
near = {oid: np.linalg.norm(tg.S - p_star[oid], axis=1) <= NEAR_RADIUS for oid in ids}
truth = {}
for o, oid in enumerate(ids):
    u1, u2, Z = to_grid(tg.S, tg.lnR_true[:, o])
    truth[oid] = RegularGridInterpolator((u2, u1), Z)


def adaptive_iterations():
    """Per run and iteration: sample class of the added sample, gap and prediction error at p_hat."""
    rows = []
    for oid in ids:
        for s in range(1, 11):
            r = load(f'kc_adaptive/kc_{oid}_seed{s:02d}.mat')
            origin, moved = np.asarray(r.samples.origin), np.asarray(r.samples.moved).astype(bool)
            lnR_fom = np.asarray(r.samples.lnR_fom)
            its = np.atleast_1d(r.iterations)
            for k, it in enumerate(its, 1):
                j = np.nonzero(origin == k)[0]
                if j.size and not moved[j[0]]:
                    true_at_p = lnR_fom[j[0]]                 # sample added exactly at p_hat
                elif k == len(its):
                    true_at_p = r.final.lnR_fom
                else:
                    true_at_p = truth[oid](np.asarray(it.p_hat)[::-1])[0]
                cls = None
                if j.size:
                    far = np.linalg.norm(np.asarray(r.samples.S)[j[0]] - p_star[oid]) > NEAR_RADIUS
                    cls = 'moved' if moved[j[0]] else ('explore' if far else 'refine')
                rows.append(dict(oid=oid, k=k, cls=cls, gap=true_at_p - L_star[oid],
                                 pred=abs(it.lnR_mean - true_at_p)))
    return rows


def replay_curves():
    curves = {oid: defaultdict(list) for oid in ids}
    for fn in sorted(glob.glob(str(RES / 'kc_replay/*.mat'))):
        x = load(fn.split('results/')[1])
        for rep in np.atleast_1d(x.replay):
            curves[x.objective.id][int(rep.n)].append(100 * np.median(rep.meanRelErr))
    return curves


def nonadaptive_points():
    pts = {oid: {'LHS': defaultdict(list), 'FFD': defaultdict(list)} for oid in ids}
    for fn in sorted(glob.glob(str(RES / 'kc_nonadaptive/kc_*.mat'))):
        name = fn.rsplit('/', 1)[1]
        r = load(fn.split('results/')[1])
        design = 'LHS' if name.startswith('kc_lhs') else 'FFD'
        for o, oid in enumerate(ids):
            pts[oid][design][int(r.n)].append(100 * np.median(r.testgrid.meanRelErr[near[oid]]))
    return pts


def per_iteration(rows, key, kmax):
    ks = np.arange(1, kmax + 1)
    q = np.array([np.percentile([x[key] for x in rows if x['k'] == k], [25, 50, 75]) for k in ks])
    return ks, q


def main():
    rows = adaptive_iterations()
    kmax = 30                                                   # few runs go beyond (max 44)
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=(1, 1), hspace=0.42, wspace=0.45)

    # (a) sample classes per iteration
    ax = fig.add_subplot(gs[0, :2])
    ks = np.arange(1, kmax + 1)
    counts = {c: np.array([sum(x['cls'] == c for x in rows if x['k'] == k) for k in ks])
              for c in ('explore', 'refine', 'moved')}
    total = sum(counts.values())
    bottom = np.zeros(ks.size)
    for c, color, label in (('explore', RED, r'Exploration ($> 2.5$ mm from $\mathbf{p}^*$)'),
                            ('refine', MATLAB_BLUE, r'Near-optimal ($\leq 2.5$ mm)'),
                            ('moved', '#85BAE0', 'Refinement (moved by distance check)')):
        share = np.divide(counts[c], total, out=np.zeros(ks.size), where=total > 0)
        ax.bar(ks, share, bottom=bottom, color=color, width=0.8, label=label)
        bottom += share
    ax2 = ax.twinx()
    ax2.plot(ks, total, 'k-', lw=1, label='Runs that add a sample')
    ax2.set_ylabel('Number of runs', fontsize=11)
    ax2.set_ylim(0, 58)
    ax2.set_yticks(range(0, 41, 10))
    ax.set(xlim=(0.4, kmax + 0.6), ylim=(0, 1.3), xlabel='Iteration $k$', ylabel='Share of added samples')
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title('(a) Samples added per iteration (40 runs)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', ncol=2, fontsize=8.5, framealpha=1)

    # (b) gap and prediction error at the found optimum
    ax = fig.add_subplot(gs[0, 2:])
    for key, color, label in (('gap', MATLAB_BLUE, r'True gap $\ln R(\hat{\mathbf{p}}) - \ln R^*$'),
                              ('pred', ORANGE, r'Prediction error $|\ln \hat{R}(\hat{\mathbf{p}}) - \ln R(\hat{\mathbf{p}})|$')):
        ks_, q = per_iteration(rows, key, kmax)
        q = np.maximum(q, 1e-4)                 # gap < 0 possible: p_hat is not restricted to the grid
        ax.fill_between(ks_, q[:, 0], q[:, 2], color=color, alpha=0.2, lw=0)
        ax.semilogy(ks_, q[:, 1], color=color, lw=1.6, label=label + ' (median, IQR)')
    ax.axhline(0.02, color='0.4', ls='--', lw=0.9)
    ax.text(kmax, 0.022, 'gap 0.02', ha='right', va='bottom', fontsize=8.5, color='0.3')
    ax.set(xlim=(1, kmax), ylim=(1e-4, 5), xlabel='Iteration $k$')
    ax.set_title('(b) Found optimum per iteration (40 runs)')
    ax.legend(loc='upper right', fontsize=8.5)

    # (c) error near p* vs N, per objective
    curves, pts = replay_curves(), nonadaptive_points()
    for j, oid in enumerate(ids):
        ax = fig.add_subplot(gs[1, j])
        ns = np.array(sorted(n for n, v in curves[oid].items() if len(v) >= MIN_RUNS))
        q = np.array([np.percentile(curves[oid][n], [25, 50, 75]) for n in ns])
        ax.fill_between(ns, q[:, 0], q[:, 2], color=MATLAB_BLUE, alpha=0.2, lw=0)
        ax.semilogy(ns, q[:, 1], color=MATLAB_BLUE, lw=1.6, label='Adaptive (replay)')
        for design, color, marker in (('LHS', ORANGE, 'o'), ('FFD', BLUE, 's')):
            d = pts[oid][design]
            n = np.array(sorted(d))
            med = np.array([np.median(d[m]) for m in n])
            lo = np.array([np.min(d[m]) for m in n])
            hi = np.array([np.max(d[m]) for m in n])
            ax.errorbar(n, med, yerr=[med - lo, hi - med], fmt=marker, color=color, ms=4, lw=0.8,
                        capsize=2, label=f'{design} (median, range)' if design == 'LHS' else 'FFD')
        ax.set(xlim=(16, 52), ylim=(0.3, 150), xlabel='Number of samples $N$')
        if j == 0:
            ax.set_ylabel('Error 1-500 Hz near $\\mathbf{p}^*$ [%]')
            ax.legend(loc='upper right', fontsize=8)
        ax.set_title(('(c) ' if j == 0 else '') + TITLES[oid])
    save(fig, 'fig09_convergence')


def true_objectives():
    """Appendix figure: true objective of the beam (top) and of the four Kelvin-cell objectives (2 x 2), with the
    near-optimal region (gap <= 0.02), the found optima of the adaptive runs and, for the Kelvin cell, the region
    within 2.5 mm of p* in which the error near p* is evaluated."""
    import glob
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from plot_showcase import beam_reference, contour_levels
    from style import GREEN, bound_ticks

    def panel(ax, u1, u2, Z, ps, ph, title, circle):
        L = Z.min()
        ax.contour(u1, u2, Z, levels=contour_levels(Z), colors='k', linewidths=0.6, linestyles='solid')
        ax.contourf(u1, u2, Z, levels=[L - 1, L + GAP_NEAR], colors=['#cfe8cf'])
        ax.contour(u1, u2, Z, levels=[L + GAP_NEAR], colors='green', linewidths=1.2, linestyles='solid')
        if circle:
            ax.add_patch(plt.Circle(ps, NEAR_RADIUS, fill=False, ls='--', color='0.3', lw=0.9))
        ax.plot(ph[:, 0], ph[:, 1], 'o', color=MATLAB_BLUE, ms=3.5, mec='k', mew=0.4, zorder=4, clip_on=False)
        ax.plot(*ps, 'D', color=GREEN, mec='k', mew=0.4, ms=6, zorder=5, clip_on=False)
        bound_ticks(ax, u1[0], u1[-1], axis='x')
        bound_ticks(ax, u2[0], u2[-1], axis='y')
        ax.set(xlabel='$p_1$', ylabel='$p_2$', aspect='equal', title=title)

    fig = plt.figure(figsize=(9.5, 14.4))
    gs = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.6, bottom=0.1, top=0.97)
    u1, u2, Z, ps = beam_reference()
    ph = np.array([np.ravel(load(f.split('results/')[1]).final.p_hat)
                   for f in sorted(glob.glob(str(RES / 'beam_adaptive/*.mat')))])
    panel(fig.add_subplot(gs[0, 1:3]), u1, u2, Z, ps, ph, 'Timoshenko beam, 50-100 Hz, $k = 1$', False)
    for j, (o, oid) in enumerate(enumerate(ids)):
        k1, k2, Zk = to_grid(tg.S, tg.lnR_true[:, o])
        ph = np.array([np.ravel(load(f'kc_adaptive/kc_{oid}_seed{s:02d}.mat').final.p_hat) for s in range(1, 11)])
        panel(fig.add_subplot(gs[1 + j // 2, 2 * (j % 2):2 * (j % 2) + 2]), k1, k2, Zk, p_star[oid], ph,
              'Kelvin cell, ' + TITLES[oid], True)
    handles = [Line2D([], [], ls='none', marker='D', color=GREEN, mec='k', label=r'Optimum $\mathbf{p}^*$'),
               Line2D([], [], ls='none', marker='o', color=MATLAB_BLUE, mec='k',
                      label=r'Found optima $\hat{\mathbf{p}}$ (beam: 50 runs, Kelvin cell: 10 runs)'),
               Patch(fc='#cfe8cf', ec='green', label=r'$\ln R \leq \ln R^* + 0.02$'),
               Line2D([], [], ls='--', color='0.3', label=r'2.5 mm around $\mathbf{p}^*$ (Kelvin cell)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.05))
    save(fig, 'fig13_true_objectives')

if __name__ == '__main__':
    main()
    true_objectives()
