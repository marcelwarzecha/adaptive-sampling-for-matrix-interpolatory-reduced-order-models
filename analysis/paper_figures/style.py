"""Plot style of the paper's MATLAB figures (Computer Modern, parula, marker colours) and shared helpers."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results'
# Any Kelvin cell result with a test-grid evaluation holds the true ln R of all objectives on the 51x51 grid
REFERENCE = 'kc_showcase/kc_f300-400_k1_seed30.mat'
OUT = Path(__file__).resolve().parent / 'figures'

PARULA = LinearSegmentedColormap.from_list('parula', [
    (0.2422, 0.1504, 0.6603), (0.2803, 0.2782, 0.9221), (0.2440, 0.4358, 0.9988),
    (0.1540, 0.5902, 0.9218), (0.0297, 0.7082, 0.8163), (0.1938, 0.7758, 0.6251),
    (0.5044, 0.7993, 0.3480), (0.8634, 0.7406, 0.1596), (0.9769, 0.9839, 0.0805)])
RED, BLUE, GREEN = (0.92, 0.18, 0.12), (0.0, 0.0, 1.0), (0.45, 0.93, 0.25)
ORANGE = (0.85, 0.325, 0.098)             # MATLAB default orange (posterior-mean surface)
MATLAB_BLUE = (0.0, 0.447, 0.741)
ASTERISK = (6, 2, 0)                      # MATLAB '*' (matplotlib '*' is a filled star)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
                     'font.size': 11, 'axes.labelsize': 15, 'axes.titlesize': 11})

# Result folder -> run_all queue that produces it
QUEUE = {'beam_reference': 'beam', 'beam_showcase': 'beam', 'beam_adaptive': 'beam', 'kc_showcase': 'kelvin_cell',
         'kc_adaptive': 'kelvin_cell', 'kc_nonadaptive': 'nonadaptive', 'kc_fd': 'fd', 'kc_replay': 'replay'}


def require(*groups):
    """Stop with a clear message if the results of these job groups are missing."""
    missing = [g for g in groups if not any((RES / g).glob('*.mat'))]
    if missing:
        queues = sorted({QUEUE[g] for g in missing})
        raise SystemExit(f'Missing results in {", ".join("results/" + g for g in missing)}. '
                         f'Run {" and ".join(f"run_all({chr(39)}{q}{chr(39)})" for q in queues)} in MATLAB first '
                         f'(see README).')


def load(rel):
    return sio.loadmat(RES / rel, squeeze_me=True, struct_as_record=False)['res']


def to_grid(points, values):
    """Scattered values on a tensor grid -> (u1, u2, Z) with Z[i2, i1]."""
    u1, u2 = np.unique(points[:, 0]), np.unique(points[:, 1])
    Z = np.full((u2.size, u1.size), np.nan)
    Z[np.searchsorted(u2, points[:, 1]), np.searchsorted(u1, points[:, 0])] = values
    return u1, u2, Z


def bound_ticks(ax, lo, hi, n=5, axis='both'):
    """MATLAB-like ticks: n equally spaced values between the bounds, rounded to 4 digits."""
    t = np.linspace(lo, hi, n)
    labels = [f'{np.round(v + 1e-12, 4):g}' for v in t]
    if axis in ('x', 'both'):
        ax.set_xticks(t, labels)
    if axis in ('y', 'both'):
        ax.set_yticks(t, labels)


def save(fig, name, exts=('pdf', 'png'), dpi=200):
    """Save to figures/<name>.<ext>; name may contain subfolders."""
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
