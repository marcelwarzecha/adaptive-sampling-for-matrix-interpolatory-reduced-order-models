"""Computational cost of the adaptive Kelvin-cell runs against non-adaptive designs and FD optimization.

fig11_cost (Fig. 11):
  (a) wall-clock time per objective, split into subroutines (mean over the runs), with the number of
      full-order evaluations: adaptive runs (kc_adaptive, 40 runs); LHS with the same N and seed as each
      adaptive run; FFD 6x6; fmincon with finite differences on the FOM and on a local ROM (kc_fd, 20 runs
      each). Non-adaptive: one pROM build plus the optimization of this objective; the build would serve
      all objectives.
  (b) adaptive runs: time per iteration and subroutine (mean over the runs that are still running).

Times are wall-clock with 12 workers, logging excluded. Non-adaptive designs assemble their
FOMs in parallel; the adaptive loop adds one FOM per iteration.

    python plot_cost.py                   # writes figures/fig11_cost.{pdf,png}
"""
import glob

import matplotlib.pyplot as plt
import numpy as np

from style import REFERENCE, RES, load, require, save

TUM = ['#000953', '#005199', '#4283B0', '#85BAE0', '#B4DCFF']
PARTS = [('fom', 'FOM assembly'), ('irka', 'Local reduction (IRKA)'), ('reproj', 'Reprojection'),
         ('train', 'SBR training'), ('opt', 'Optimization')]
COLORS = dict(zip([p for p, _ in PARTS], TUM))
FD_COLOR = '0.55'

require('kc_showcase', 'kc_adaptive', 'kc_nonadaptive', 'kc_fd')
ids = list(load(REFERENCE).testgrid.objective_ids)


def adaptive_runs():
    runs = []
    for oid in ids:
        for s in range(1, 11):
            r = load(f'kc_adaptive/kc_{oid}_seed{s:02d}.mat')
            t = r.time
            runs.append(dict(oid=oid, seed=s, N=int(r.final.n_total), t=t,
                             fom=t.initial_sampling + np.sum(t.sample_adding), irka=np.sum(t.local_reduction),
                             reproj=np.sum(t.reprojection), train=np.sum(t.training), opt=np.sum(t.optimization)))
    return runs


def nonadaptive(r, oid):
    op = [o for o in np.atleast_1d(r.optimization) if o.objective_id == oid][0]
    t = r.time
    return dict(N=int(r.n), fom=t.initial_sampling, irka=t.local_reduction, reproj=t.reprojection,
                train=t.training, opt=op.time)


def bars(runs):
    out = [('Adaptive', runs, 'N')]
    paired = [f"kc_nonadaptive/kc_lhs{x['N']}_seed{x['seed']:02d}.mat" for x in runs]
    out.append(('LHS, same $N$', [nonadaptive(load(f), x['oid']) for f, x in zip(paired, runs)], 'N'))
    r = load('kc_nonadaptive/kc_ffd6x6.mat')
    out.append((r'FFD 6$\times$6', [nonadaptive(r, oid) for oid in ids], 'N'))
    for level in ('ROM', 'FOM'):
        rows = []
        for fn in sorted(glob.glob(str(RES / f'kc_fd/kc_fd_{level}_*.mat'))):
            r = load(fn.split('results/')[1])
            rows.append(dict(N=int(r.fmincon_funccount), total=r.time.total))
        out.append((f'FD on {level}', rows, 'evals'))
    return out


def main():
    runs = adaptive_runs()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=(1.05, 1), wspace=0.32))

    # (a) time per objective
    data = bars(runs)
    y = np.arange(len(data))[::-1]
    for yi, (lab, rows, kind) in zip(y, data):
        n = np.median([x['N'] for x in rows])
        if 'total' in rows[0]:
            total = np.mean([x['total'] for x in rows])
            ax.barh(yi, total, color=FD_COLOR, edgecolor='k', lw=0.5, height=0.62)
            note = f'{n:.0f} FOM evaluations' if 'FOM' in lab else f'{n:.0f} FOM + IRKA evaluations'
        else:
            left = 0.0
            for part, _ in PARTS:
                v = np.mean([x[part] for x in rows])
                ax.barh(yi, v, left=left, color=COLORS[part], edgecolor='k', lw=0.5, height=0.62)
                left += v
            total = left
            note = f'$N = {n:.0f}$ FOMs' + (' (median)' if lab == 'Adaptive' else '')
        ax.text(total + 25, yi, f'{total:.0f} s, {note}', va='center', fontsize=9)
    ax.set_yticks(y, [d[0] for d in data])
    ax.set(xlim=(0, 2750), xlabel='Wall-clock time per objective [s]')
    ax.set_xticks(range(0, 2501, 500))
    ax.set_title('(a) Time per objective (mean over the runs)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[p], ec='k', lw=0.5) for p, _ in PARTS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=FD_COLOR, ec='k', lw=0.5))
    ax.legend(handles, [l for _, l in PARTS] + ['fmincon with finite differences'], loc='center right', bbox_to_anchor=(1, 0.58), fontsize=8.5)

    # (b) adaptive: time per iteration
    kmax = 30
    ks = np.arange(1, kmax + 1)
    per = {p: np.zeros(kmax) for p, _ in PARTS}
    for k in ks:
        active = [x for x in runs if len(np.atleast_1d(x['t'].optimization)) >= k]
        for x in active:
            t = x['t']
            sa, lr = np.atleast_1d(t.sample_adding), np.atleast_1d(t.local_reduction)
            per['fom'][k - 1] += (t.initial_sampling if k == 1 else sa[k - 2]) / len(active)
            per['irka'][k - 1] += lr[k - 1] / len(active)
            per['reproj'][k - 1] += np.atleast_1d(t.reprojection)[k - 1] / len(active)
            per['train'][k - 1] += np.atleast_1d(t.training)[k - 1] / len(active)
            per['opt'][k - 1] += np.atleast_1d(t.optimization)[k - 1] / len(active)
    bottom = np.zeros(kmax)
    for p, lab in PARTS:
        bx.bar(ks, per[p], bottom=bottom, color=COLORS[p], edgecolor='k', lw=0.3, width=0.8, label=lab)
        bottom += per[p]
    bx.set(xlim=(0.4, kmax + 0.6), xlabel='Iteration $k$', ylabel='Time per iteration [s]')
    bx.set_title('(b) Adaptive runs: time per iteration (mean)')
    bx.set_ylim(0, 1.3 * bottom.max())
    bx.legend(loc='upper left', ncol=2, fontsize=8.5)
    save(fig, 'fig11_cost')

    total = np.mean([x['t'].total for x in runs])
    print('adaptive shares of the mean total:', {p: f"{np.mean([x[p] for x in runs]) / total:.1%}" for p, _ in PARTS})


if __name__ == '__main__':
    main()
