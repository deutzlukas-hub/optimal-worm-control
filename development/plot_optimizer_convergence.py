# From built-in
from argparse import ArgumentParser
from typing import List
import matplotlib.pyplot as plt
# From third-party
import numpy as np
# From my projects
from worm_experiments.util import load_sweep
from optimal_worm_control.util import linear_norm
# From waveform-optimizer
from optimal_worm_control.waveform_optimizer import WaveformOptimizer
from optimal_worm_control.dirs import test_data_dir, development_dir


def waveform(s_arr: np.ndarray, p_arr: List[float]):
    c0, lam0 = p_arr[0], p_arr[1]
    q0 = 2.0 * np.pi / lam0
    A0 = c0 * q0
    return A0 * np.sin(q0 * s_arr)

def plot_optimize_sinusoidal_waveform_spsa():

    parser = ArgumentParser()
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=50)
    cmd_args = parser.parse_args()

    mu = 1e-3

    args = [
        '--dt', str(0.01),
        '--N', str(250),
        '--dimensionless_from_physical', str(True),
        '--mu', str(mu),
        '--no-minimize',
        '--normalize',
    ]

    WFO = WaveformOptimizer(args)
    WFO.set_objective('speed')
    WFO.set_normalizer(linear_norm(0.0, 0.2))
    p0 = [1.0, 1.0]
    history = WFO.optimize_waveform_parameter_spsa(waveform, p0, cmd_args.epsilon, cmd_args.K, max_iter=cmd_args.max_iter)
    p_arr_history = np.array([tup[0] for tup in history])

    h5_filename = 'sweep_celegans_mu_min=-3.0_mu_max=1.0_mu_step=0.2_c0_min=0.5_c0_max=2.0_c0_step=0.1lam0_min=0.5_lam0_max=2.0_lam0_step=0.1_N=750_dt=0.001.h5'
    h5, PG = load_sweep(test_data_dir / h5_filename)
    c0_arr, lam0_arr = PG.v_from_key('c0'), PG.v_from_key('lam0')
    lam0_grid, c0_grid = np.meshgrid(lam0_arr, c0_arr)

    mu_idx = np.abs(mu - PG.v_from_key('mu')).argmin()
    U = h5['U'][mu_idx, :]

    levels = np.arange(0, 1.01, 0.2)
    ax = plt.subplot(111)
    CS = ax.contourf(lam0_grid, c0_grid, U / U.max(), cmap='Blues', levels=levels, extend='max')
    ax.contour(lam0_grid, c0_grid, U / U.max(), colors='k', levels=levels, linestyles='-')
    cbar = plt.colorbar(CS, ax=ax)
    cbar.set_label(r'$U/U_{\max}$')

    ax.plot(p_arr_history[:, 1], p_arr_history[:, 0], '-o')
    plt.savefig(development_dir / 'optimize_U.svg')


if __name__ == '__main__':
    plot_optimize_sinusoidal_waveform_spsa()