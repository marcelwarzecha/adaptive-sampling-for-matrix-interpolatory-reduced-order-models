# Adaptive sampling for matrix-interpolatory reduced-order models

MATLAB code for the paper

> M. Warzecha, S. Resch-Schopper, G. Müller: *Uncertainty-Aware Calculation of Analytical Gradients of Matrix-Interpolatory Reduced-Order Models for Sample-Efficient Structural Optimization*, Journal of Probability and Statistics.

The framework builds a parametric reduced-order model (pROM) by matrix interpolation and optimizes the frequency response of a structure with it:

1. Full-order models (FOMs) at the samples are reduced locally with IRKA and projected onto one global basis.
2. Every entry of the reduced matrices is regressed over the parameter space with sparse Bayesian regression (SBR) on a total-degree monomial basis.
3. One Thompson realization of the regression coefficients is drawn per iteration. The resulting pROM is optimized with `fmincon`, using analytical gradients from the adjoint method.
4. The found optimum becomes the next FOM sample, and the loop repeats until the optimum value has stabilized.

Two examples are included: a Timoshenko beam (2,400 DOFs) and a Kelvin cell (10,728 DOFs), each with two geometric parameters.

## Requirements

- MATLAB R2022a or newer, with the Statistics and Machine Learning, Optimization and Parallel Computing toolboxes. Without the Parallel Computing Toolbox, everything runs serially (`'workers', 0`), only more slowly.
- For the figures and tables: Python 3 with NumPy, SciPy and Matplotlib.

No further installation is needed. The finite element routines for the Kelvin cell are included in `external/bm-mfem`, and the full-order reference data in `data/`.

## Quick start

```matlab
cd <path-to-repository>
run_all('smoke')                  % ~15-30 min: checks the installation
run_all('example')                % ~1 h: one beam run, one Kelvin cell run, two non-adaptive pROMs
```

`run_all('smoke')` checks the toolboxes, the model assembly and the reference data. It compares the adjoint gradient with direct sensitivities and with finite differences, and it runs every job type once with reduced settings. Afterwards, `results/smoke/failed.txt` must be empty, and the line `[check_gradient …] … -> true` must appear in `results/smoke/gradient_*.log` for both models.

Options for every queue:

```matlab
run_all('example', 'workers', 4)          % parallel pool size (default 12; 0 = serial)
run_all('example', 'dry_run', true)       % list the jobs and their status only
run_all('beam', 'groups', {'beam_showcase'})   % only some job groups of a queue
```

Every job writes `results/<group>/<id>.mat` and a `.log` file. Finished jobs are skipped, so after an interruption, simply call `run_all` again. A failing job writes `<id>.error.txt`, the queue continues, and the job is retried on the next call. At the end, `run_all` writes CSV summaries to `results/`.

## Reproducing the paper

```matlab
run_all('paper')
```

This runs the following queues in order. Each of them can also be run on its own. The times were measured with 12 workers.

| Queue | Jobs | Time |
|---|---|---|
| `beam` | beam reference grid (51×51 FOMs), representative run (seed 30), adaptive runs with seeds 1–50 | ~20 min |
| `kelvin_cell` | representative run (300–400 Hz, k = 1, seed 30), adaptive runs for 4 objectives × seeds 1–10 | ~8 h |
| `nonadaptive` | square FFDs (5², 6², 7², 10²), LHS with N = 30, 32, 35, 40, 44 × seeds 1–10, and for every adaptive run the LHS with its final N and seed (needs `kelvin_cell`) | ~9 h |
| `fd` | `fmincon` with finite differences on the FOM and on non-parametric ROMs, 4 objectives × seeds 1–5 | ~15 h |
| `replay` | posterior-mean pROM of every adaptive Kelvin cell run after each iteration, evaluated near the true optimum (needs `kelvin_cell`) | ~7 h |

All settings are in `config/production_config.m`, and the jobs and seeds of every queue are in `config/build_queue.m`. Every run initializes the random number generator with `rng(seed, 'twister')`. Wall-clock times and results in the last digits can differ between machines and MATLAB versions.

Then create the figures and tables:

```bash
cd analysis/paper_figures
python3 plot_showcase.py       # Figs. 4, 5, 8, 12
python3 plot_error_maps.py     # Figs. 7, 10
python3 plot_convergence.py    # Figs. 9, 13
python3 plot_cost.py           # Fig. 11
cd ../results_summary
python3 summarize.py           # Tables 6-9 and the statistics in the text (summary.md)
```

The figures are written to `analysis/paper_figures/figures/`. After `run_all('example')`, `plot_showcase.py` and `plot_error_maps.py` can already be run. The other scripts need the results of `run_all('paper')`; if results are missing, they say which queue produces them.

## Layout

```
run_all.m                    entry point (queue runner, resumable)
setup_paths.m                adds code/, external/, src/ and config/ to the path
config/production_config.m   settings of all runs
config/build_queue.m         jobs of every queue
src/                         drivers (run_adaptive, run_nonadaptive, run_fd_comparison,
                             run_reference_grid, replay_adaptive, check_gradient, ...)
code/core/                   framework: models, MOR (IRKA, global basis), regression (SBR,
                             compiled pROM), objective and adjoint gradient
external/bm-mfem/            finite element routines for the Kelvin cell
data/                        full-order reference data of the Kelvin cell (51×51 grid)
analysis/                    Python scripts for the figures and tables
```

## Result files

Every result file contains one struct `res`:

- `run_adaptive`:
  - `res.samples`: `S` (samples in the order they were added), `origin` (0 = initial, k = added after iteration k), `moved` (moved by the distance check), `lnR_fom` (true objective at the sample)
  - `res.iterations(k)`: `n_samples`, `r`, `poly_order`, `z0` (starting point), `p_hat` and `lnR_ts` (found optimum and its value on the Thompson-sampled pROM), `lnR_mean` (posterior-mean pROM at `p_hat`), `eps_step` and `eps` (stopping criterion), `fmincon_*`, `min_eig_*`/`max_eig_*` (smallest/largest eigenvalue of the predicted K, D, M for the Thompson draw and the posterior mean at `pd_points`, first row = found optimum), `distance_check`, optionally `surface_points`, `surface_lnR_ts`, `surface_lnR_mean`
  - `res.time`: wall-clock time per subroutine and iteration; `res.time.total` excludes logging
  - `res.final`: number of samples and iterations, final optimum and its true objective value
  - `res.testgrid` (Kelvin cell): error of the final posterior-mean pROM on the 51×51 grid, and ln R of pROM and FOM for every objective
  - `res.meta`: seed, settings, git commit, MATLAB version
- `run_nonadaptive`: `res.samples`, `res.time`, `res.optimization(o)` for every objective, `res.testgrid`, `res.meta`
- `run_fd_comparison`: optimum, ln R (ROM and FOM), `fmincon` statistics, runtime, `res.meta`
- `replay_adaptive`: `res.replay(j)` with the errors of the posterior-mean pROM after iteration `res.replay(j).iteration` near the true optimum

## Third-party code

- `code/core/SB1_Release_110`: SparseBayes baseline implementation V1.10 by Michael E. Tipping (sparse Bayesian regression), with adaptations for this framework (marked in `SB1_Estimate.m`). Copyright 2009 Michael E. Tipping, GPL-2.0 ([miketipping.com/sparsebayes.htm](https://miketipping.com/sparsebayes.htm)).
- `code/core/Toolboxes/polyfitn`: `polyfitn` by John D'Errico. Copyright 2016 John D'Errico, BSD-2-Clause (see `license.txt` there).
- `external/bm-mfem`: subset of the finite element code bm-mfem of the Chair of Structural Mechanics, Technical University of Munich, by Dr. Quirin Aumann; the Timoshenko beam element by Sebastian Resch-Schopper. Released here under GPL-2.0 with permission.

## Citation

If you use this code, please cite the paper above.

## Licence

Copyright (c) 2026 Marcel Warzecha, Sebastian Resch-Schopper.

This code is distributed under the GNU General Public License, version 2 (see `LICENSE`), the licence of the included SparseBayes code. The third-party parts listed above keep the copyright of their authors; `polyfitn` remains under its BSD licence, which is compatible with GPL-2.0.
