# Data

| File | Content |
|---|---|
| `FFD_51x51_1000Hz.mat` | Full-order transfer functions of the Kelvin cell on a 51×51 grid over l_x ∈ [0.055, 0.080] m, l_y ∈ [0.020, 0.045] m, at 1–1000 Hz (1 Hz steps). Variables `S` (2601×2, m), `omegaHz` (1×1000), `y` (1×1000×2601, complex). MD5 `58381687c11acd4d5b912135e49bc8b8`. |

This grid is the reference ("true") solution for all Kelvin cell error maps and for the true optimum of every objective (2,601 full-order solves). It agrees with the full-order model assembled by `code/core/models/KelvinCell.m` (10,728 DOFs).

All other full-order models are assembled during the runs. The beam reference grid is computed by `run_all('beam')` (job `beam_reference`, about 10 minutes).
