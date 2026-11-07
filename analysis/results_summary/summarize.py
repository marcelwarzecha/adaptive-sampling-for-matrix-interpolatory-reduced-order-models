"""The numbers of the paper (Tables 6-9 and the statistics in the text).

Kelvin cell: adaptive runs (4 objectives x 10 seeds) vs FFD 5x5/6x6/7x7/10x10, LHS 30/35/40/44 (10 seeds) and
fmincon with finite differences on the FOM / a local ROM (5 seeds). Pooled over the objectives and per objective.
Equal budget: every adaptive run is paired with the LHS of the same N and seed ('LHS, same N') and with the square
FFD grid closest to its N ('FFD, closest N': 5x5, 6x6 or 7x7).
The paired table counts, per metric, the runs in which the adaptive pROM is better than its LHS partner.
Iteration history of the representative Kelvin cell run (seed 30).
Beam: the 50 adaptive runs.

Definitions
- p*, ln R*: minimum of the true ln R on the 51x51 reference grid. gap = true ln R(p_hat) - ln R* (can be slightly
  negative, since p_hat is not restricted to the grid).
- pROM errors: relative transfer-function error of the posterior-mean pROM over 1-500 Hz ('1-500 Hz', the paper's
  error maps) and within the objective band ('band'); 'at p*' = grid point of p*, 'near p*' = median over the grid
  points within 2.5 mm, 'global' = median over the grid.
- wrong pROM minimum: the pROM's minimum on the grid has a true gap > 0.02.
- Exploration samples (adaptive): added samples more than 2.5 mm from p* (the radius of 'near p*') that were not
  moved by the distance check (moved samples lie next to an existing sample near the optimum). Their time share
  counts, per sample, the iteration that chose it (reprojection, training, optimization) plus its FOM assembly and IRKA.
- Indefinite: smallest eigenvalue of the predicted K or M at p_hat < 0 (Thompson draw or posterior mean).
- Times: wall-clock with 12 workers, logging excluded. Non-adaptive: one pROM build plus the
  optimization for this objective (the build serves all objectives).

    python summarize.py            # prints Markdown and writes summary.md next to this script
"""
import glob
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / 'results'
NEAR_RADIUS = 2.5e-3
GAP_FAIL = 0.02


def load(path):
    return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)['res']


missing = [g for g in ('kc_showcase', 'kc_adaptive', 'kc_nonadaptive', 'kc_fd', 'beam_reference', 'beam_adaptive')
           if not any((RES / g).glob('*.mat'))]
if missing:
    raise SystemExit(f'Missing results in {", ".join("results/" + g for g in missing)}. '
                     "Run run_all('paper') in MATLAB first (see README).")
ref_tg = load(RES / 'kc_showcase/kc_f300-400_k1_seed30.mat').testgrid    # true ln R of all objectives
S, ids, lnR_true = ref_tg.S, list(ref_tg.objective_ids), ref_tg.lnR_true
p_star = {oid: S[np.argmin(lnR_true[:, o])] for o, oid in enumerate(ids)}
L_star = {oid: lnR_true[:, o].min() for o, oid in enumerate(ids)}
near = {oid: np.linalg.norm(S - p_star[oid], axis=1) <= NEAR_RADIUS for oid in ids}
at = {oid: np.argmin(np.linalg.norm(S - p_star[oid], axis=1)) for oid in ids}


def accuracy(tg, oid):
    o = ids.index(oid)
    i_min = np.argmin(tg.lnR_prom[:, o])
    return dict(m_at=100 * tg.meanRelErr[at[oid]], m_near=100 * np.median(tg.meanRelErr[near[oid]]),
                m_glob=100 * np.median(tg.meanRelErr),
                b_at=100 * tg.bandRelErr[at[oid], o], b_near=100 * np.median(tg.bandRelErr[near[oid], o]),
                b_glob=100 * np.median(tg.bandRelErr[:, o]),
                wrong_min=lnR_true[i_min, o] - L_star[oid] > GAP_FAIL)


def optimum(oid, p_hat, lnR_fom, samples):
    return dict(gap=lnR_fom - L_star[oid], dist=1e3 * np.linalg.norm(p_hat - p_star[oid]),
                n_near=int((np.linalg.norm(samples - p_star[oid], axis=1) <= NEAR_RADIUS).sum()))


def adaptive_run(r, oid):
    t = r.time
    rep, tr, opt = (np.atleast_1d(x) for x in (t.reprojection, t.training, t.optimization))
    sa, lr = np.atleast_1d(t.sample_adding), np.atleast_1d(t.local_reduction)
    origin = np.asarray(r.samples.origin)
    added = origin > 0
    cost = np.array([rep[k-1] + tr[k-1] + opt[k-1] + sa[k-1] + lr[k] for k in origin[added]])
    expl = (np.linalg.norm(np.asarray(r.samples.S)[added] - p_star[oid], axis=1) > NEAR_RADIUS) \
        & ~np.asarray(r.samples.moved).astype(bool)[added]
    its = np.atleast_1d(r.iterations)
    indef = lambda field: np.array([np.any(np.atleast_2d(getattr(I, field))[0, [0, 2]] < 0) for I in its])
    ts, mean = indef('min_eig_ts'), indef('min_eig_mean')
    return dict(oid=oid, N=r.final.n_total, n_added=r.final.n_added, iters=r.final.n_iterations,
                r=r.final.r, order=r.final.poly_order, time=t.total, t_fom=t.initial_sampling + sa.sum(),
                t_irka=lr.sum(), t_train=tr.sum(), t_reproj=rep.sum(), t_opt=opt.sum(),
                pred_err=abs(r.final.lnR_mean - r.final.lnR_fom),
                n_expl=int(expl.sum()), n_expl_early=int((expl & (origin[added] <= 10)).sum()),
                t_expl=cost[expl].sum() / t.total,
                ts_indef=ts.mean(), ts_indef_early=ts[:9].mean(), ts_indef_late=ts[-5:].mean(),
                mean_indef=mean.mean(), final_indef=bool(mean[-1]),
                ts_skipped=sum(I.n_ts_skipped for I in its),
                **optimum(oid, r.final.p_hat, r.final.lnR_fom, r.samples.S), **accuracy(r.testgrid, oid))


def nonadaptive_rows(r):
    rows = []
    for op in np.atleast_1d(r.optimization):
        oid = op.objective_id
        rows.append(dict(oid=oid, N=r.n, r=r.r, order=r.poly_order,
                         time=r.time.build_total + op.time, t_opt=op.time,
                         pred_err=abs(op.lnR_mean - op.lnR_fom),
                         **optimum(oid, op.p_hat, op.lnR_fom, r.samples.S), **accuracy(r.testgrid, oid)))
    return rows


def find(name):
    fn = RES / 'kc_nonadaptive' / name
    return fn if fn.exists() else None


def ffd_grids():
    """Square FFD grids below 10x10."""
    names = {Path(f).name for f in glob.glob(str(RES / 'kc_nonadaptive/kc_ffd*x*.mat'))}
    grids = [tuple(int(v) for v in n[6:-4].split('x')) for n in names]
    return sorted(g for g in grids if g[0] == g[1] and g[0] * g[1] < 100)


def paired(adaptive):
    """LHS with the same N and seed, and the closest square FFD grid, for every adaptive run."""
    grids = ffd_grids()
    lhs, ffd = [], []
    for x in adaptive:
        rows = nonadaptive_rows(load(find(f"kc_lhs{x['N']}_seed{x['seed']:02d}.mat")))
        lhs.append(next(y for y in rows if y['oid'] == x['oid']))
        dn = min(abs(a * b - x['N']) for a, b in grids)
        for a, b in grids:
            if abs(a * b - x['N']) == dn:
                ffd.append(next(y for y in nonadaptive_rows(load(find(f'kc_ffd{a}x{b}.mat'))) if y['oid'] == x['oid']))
    return lhs, ffd


def collect():
    d = {'Adaptive': [dict(adaptive_run(load(RES / f'kc_adaptive/kc_{oid}_seed{s:02d}.mat'), oid), seed=s)
                      for oid in ids for s in range(1, 11)]}
    d['LHS, same N'], d['FFD, closest N'] = paired(d['Adaptive'])
    for lab, pattern in [('FFD 5×5', 'kc_nonadaptive/kc_ffd5x5.mat'),
                         ('FFD 6×6', 'kc_nonadaptive/kc_ffd6x6.mat'),
                         ('FFD 7×7', 'kc_nonadaptive/kc_ffd7x7.mat'),
                         ('FFD 10×10', 'kc_nonadaptive/kc_ffd10x10.mat'),
                         ('LHS30', 'kc_nonadaptive/kc_lhs30_seed*.mat'),
                         ('LHS35', 'kc_nonadaptive/kc_lhs35_seed*.mat'),
                         ('LHS40', 'kc_nonadaptive/kc_lhs40_seed*.mat'),
                         ('LHS44', 'kc_nonadaptive/kc_lhs44_seed*.mat')]:
        d[lab] = [y for fn in sorted(glob.glob(str(RES / pattern))) for y in nonadaptive_rows(load(fn))]
    for level in ('FOM', 'ROM'):
        d['FD ' + level] = []
        for fn in sorted(glob.glob(str(RES / f'kc_fd/kc_fd_{level}_*.mat'))):
            r = load(fn)
            oid = r.objective if isinstance(r.objective, str) else r.objective.id
            d['FD ' + level].append(dict(oid=oid, time=r.time.total, fevals=r.fmincon_funccount,
                                         gap=r.lnR_fom - L_star[oid], dist=1e3 * np.linalg.norm(r.p_hat - p_star[oid])))
    return d


def stat(rows, key, fmt, with_range):
    v = np.array([x[key] for x in rows if key in x], dtype=float)
    if v.size == 0:
        return '–'
    s = fmt.format(np.median(v))
    if with_range and v.size > 1 and v.min() != v.max():
        s += f' [{fmt.format(v.min())}–{fmt.format(v.max())}]'
    return s


ROWS = [('N', 'Samples N', '{:.0f}', True), ('time', 'Time per objective [s]', '{:.0f}', True),
        ('t_opt', '  of which optimization [s]', '{:.0f}', False),
        ('r', 'Global basis r', '{:.0f}', False), ('order', 'Polynomial degree', '{:.0f}', False),
        ('gap', 'gap ln R(p̂) − ln R* (median)', '{:.4f}', False), ('dist', '‖p̂ − p*‖ [mm]', '{:.2f}', False),
        ('pred_err', '|ln R̂ − ln R| at p̂', '{:.3f}', False),
        ('fevals', 'Objective evaluations (FD)', '{:.0f}', True),
        ('n_near', 'Samples within 2.5 mm of p*', '{:.0f}', False),
        ('m_at', 'Error 1–500 Hz at p* [%]', '{:.1f}', True), ('m_near', 'Error 1–500 Hz near p* [%]', '{:.1f}', True),
        ('m_glob', 'Error 1–500 Hz global [%]', '{:.1f}', False),
        ('b_at', 'Error band at p* [%]', '{:.1f}', True), ('b_near', 'Error band near p* [%]', '{:.1f}', True),
        ('b_glob', 'Error band global [%]', '{:.1f}', False)]


def table(d, title):
    labs = list(d)
    print(f'\n### {title}\n')
    print('| Metric | ' + ' | '.join(labs) + ' |')
    print('|---' * (len(labs) + 1) + '|')
    for key, name, fmt, rng in ROWS:
        print(f'| {name} | ' + ' | '.join(stat(d[l], key, fmt, rng) for l in labs) + ' |')
    count = lambda rows, f: f'{sum(f(x) for x in rows)}/{len(rows)}'
    print('| gap > 0.02 | ' + ' | '.join(count(d[l], lambda x: x['gap'] > GAP_FAIL) for l in labs) + ' |')
    print('| gap > 5e-3 | ' + ' | '.join(count(d[l], lambda x: x['gap'] > 5e-3) for l in labs) + ' |')
    print('| Largest gap | ' + ' | '.join(f"{max(x['gap'] for x in d[l]):.3f}" for l in labs) + ' |')
    print('| pROM minimum in wrong place | ' + ' | '.join(
        count(d[l], lambda x: x['wrong_min']) if 'wrong_min' in d[l][0] else '–' for l in labs) + ' |')


def paired_table(ad, lhs, title):
    print(f'\n### {title}\n')
    print('| Metric | Adaptive | LHS, same N | Ratio LHS/adaptive (median) | Adaptive better |')
    print('|---|---|---|---|---|')
    for key, name, fmt in [('gap', 'gap ln R(p̂) − ln R*', '{:.4f}'), ('dist', '‖p̂ − p*‖ [mm]', '{:.2f}'),
                           ('m_at', 'Error 1–500 Hz at p* [%]', '{:.1f}'),
                           ('m_near', 'Error 1–500 Hz near p* [%]', '{:.1f}'),
                           ('b_at', 'Error band at p* [%]', '{:.1f}'), ('b_near', 'Error band near p* [%]', '{:.1f}'),
                           ('m_glob', 'Error 1–500 Hz global [%]', '{:.1f}'), ('b_glob', 'Error band global [%]', '{:.1f}')]:
        a, b = np.array([x[key] for x in ad]), np.array([x[key] for x in lhs])
        ratio = '–' if key == 'gap' else f'{np.median(b / a):.1f}'
        print(f'| {name} | {stat(ad, key, fmt, False)} | {stat(lhs, key, fmt, False)} | {ratio} | '
              f'{(a < b).sum()}/{a.size} |')


def adaptive_details(rows, title):
    print(f'\n### {title}\n')
    print('| Metric | median [range] |')
    print('|---|---|')
    for key, name, fmt in [('n_added', 'Added samples', '{:.0f}'), ('iters', 'Iterations', '{:.0f}'),
                           ('t_fom', 'Time: FOM assembly [s]', '{:.0f}'), ('t_train', 'Time: SBR training [s]', '{:.0f}'),
                           ('t_irka', 'Time: IRKA [s]', '{:.0f}'), ('t_opt', 'Time: optimization [s]', '{:.0f}'),
                           ('t_reproj', 'Time: reprojection [s]', '{:.0f}'),
                           ('n_expl', 'Exploration samples', '{:.0f}'),
                           ('n_expl_early', '  of which from iterations 1–10', '{:.0f}'),
                           ('t_expl', 'Time share of exploration samples', '{:.2f}'),
                           ('ts_indef', 'Thompson draw indefinite at p̂ (share of iterations)', '{:.2f}'),
                           ('ts_indef_early', '  iterations 1–9', '{:.2f}'), ('ts_indef_late', '  last 5 iterations', '{:.2f}'),
                           ('mean_indef', 'Posterior mean indefinite at p̂ (share of iterations)', '{:.2f}')]:
        print(f'| {name} | {stat(rows, key, fmt, True)} |')
    print(f"| Final pROM indefinite at p̂ | {sum(x['final_indef'] for x in rows)}/{len(rows)} |")
    print(f"| Thompson draws skipped (no posterior covariance) | {sum(x['ts_skipped'] for x in rows)} |")


def kc_iteration_history():
    """Iteration history of the representative Kelvin cell run (300-400 Hz, k = 1, seed 30)."""
    r = load(RES / 'kc_showcase/kc_f300-400_k1_seed30.mat')
    origin, moved = np.asarray(r.samples.origin), np.asarray(r.samples.moved).astype(bool)
    lnR = np.asarray(r.samples.lnR_fom)
    print('\n### Kelvin cell, representative run (300-400 Hz, k = 1, seed 30): iteration history\n')
    print('| k | N | q | r | Found optimum [mm] | Surrogate optimum ln R | ln R of the added sample | |')
    print('|---|---|---|---|---|---|---|---|')
    for it in np.atleast_1d(r.iterations):
        k = int(it.iteration)
        j = np.nonzero(origin == k)[0]
        added = f'{lnR[j[0]]:.4f}' if j.size else '--'
        mark = 'm' if j.size and moved[j[0]] else ''
        p = ', '.join(f'{1e3 * v:.2f}' for v in np.ravel(it.p_hat))
        print(f'| {k} | {int(it.n_samples)} | {int(it.poly_order)} | {int(it.r)} | [{p}] | {it.lnR_ts:.4f} | {added} | {mark} |')


def beam():
    br = load(glob.glob(str(RES / 'beam_reference/*.mat'))[0])
    L, ps = np.min(br.lnR), np.ravel(br.p_star)
    runs = [load(f) for f in sorted(glob.glob(str(RES / 'beam_adaptive/*.mat')))]
    rows = [dict(N=b.final.n_total, time=b.time.total, gap=b.final.lnR_fom - L,
                 dist=1e3 * np.linalg.norm(b.final.p_hat - ps)) for b in runs]
    print(f'\n### Beam, {len(rows)} adaptive runs\n')
    print('| Samples N | Time [s] | gap > 5e-3 | gap > 0.02 | Largest gap | ‖p̂ − p*‖ [mm] |')
    print('|---|---|---|---|---|---|')
    print(f"| {stat(rows, 'N', '{:.0f}', True)} | {stat(rows, 'time', '{:.1f}', True)} | "
          f"{sum(x['gap'] > 5e-3 for x in rows)}/{len(rows)} | {sum(x['gap'] > GAP_FAIL for x in rows)}/{len(rows)} | "
          f"{max(x['gap'] for x in rows):.4f} | {stat(rows, 'dist', '{:.3f}', False)} |")


def main():
    d = collect()
    table(d, 'Kelvin cell, all objectives')
    paired_table(d['Adaptive'], d['LHS, same N'], 'Kelvin cell, paired: each adaptive run vs the LHS with its N and seed')
    adaptive_details(d['Adaptive'], 'Kelvin cell, adaptive runs: cost, exploration, definiteness')
    for oid in ids:
        sub = {lab: [x for x in rows if x['oid'] == oid] for lab, rows in d.items()}
        table(sub, f'Kelvin cell, {oid} (p* = ({1e3 * p_star[oid][0]:.1f}, {1e3 * p_star[oid][1]:.1f}) mm, '
                   f'ln R* = {L_star[oid]:.3f})')
        paired_table(sub['Adaptive'], sub['LHS, same N'], f'{oid}, paired with the LHS of the same N and seed')
        adaptive_details(sub['Adaptive'], f'{oid}, adaptive runs')
    kc_iteration_history()
    beam()


if __name__ == '__main__':
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    print(out)
    (HERE / 'summary.md').write_text('# Results (generated by summarize.py)\n' + out)
