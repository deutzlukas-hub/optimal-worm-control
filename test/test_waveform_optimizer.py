# From built-in
from pathlib import Path
from typing import List
from unittest import TestCase
import h5py
import matplotlib.pyplot as plt
# From third-party
import numpy as np
# From my projects
from parameter_scan import ParameterGrid
from scipy.interpolate import RectBivariateSpline
from worm_experiments.sweeper import Sweeper
from worm_experiments.planar_undulation.planar_undulation_post_processor import PostProcessor
from worm_experiments.util import parameters_from_args, flatten_parameters
from worm_experiments.planar_undulation.planar_undulation_parameter import planar_undulation_parser
from worm_experiments.planar_undulation.sinusoidal_traveling_wave import SinusoidalTravelingWave
from worm_experiments.util import load_sweep
from worm_rod_engine.visualize.plot import plot_scalar_field

from optimal_worm_control.util import linear_norm
# From waveform-optimizer
from optimal_worm_control.waveform_optimizer import WaveformOptimizer
from optimal_worm_control.dirs import test_data_dir

def waveform(s_arr: np.ndarray, p_arr: List[float]):
    c0, lam0 = p_arr[0], p_arr[1]
    q0 = 2.0 * np.pi / lam0
    A0 = c0 * q0
    return A0 * np.sin(q0 * s_arr)

class TestWaveFormOptimize(TestCase):

    def plot_generate_curvature(self):

        args = [
            '--dt', str(0.01),
            '--N', str(250),
            '--dimensionless_from_physical', str(True),
            '--mu', str(1e-3)
        ]

        WFO = WaveformOptimizer(args)
        WFO.waveform = waveform
        k0_arr = WFO._generate_input_curvature([1.0, 1.0])

        ax = plt.subplot(111)
        plot_scalar_field(ax, k0_arr, extent=[0,1,0, WFO.params['experiment_param'].T_sim])


    def plot_solver(self):

        args = [
            '--dt', str(0.01),
            '--N', str(250),
            '--dimensionless_from_physical', str(True),
            '--mu', str(1e-3)
        ]

        WFO = WaveformOptimizer(args)
        WFO.waveform = waveform
        output = WFO._solve([1.0, 1.0])
        r = output['FS'].r
        R = r.mean(axis=-1)
        r_mp = r[:, :, r.shape[-1]//2]

        plt.plot(r_mp[:, 0], r_mp[:, 1])
        plt.plot(R[:, 0], R[:, 1])
        plt.show()

    def plot_optimization(self):

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
        history = WFO.optimize_waveform_parameter_spsa(waveform, p0, 0.1, 2, max_iter=10)
        p_arr = np.array([tup[0] for tup in history])
        U = [tup[1] for tup in history]

        gs = plt.GridSpec(3, 1)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        ax0.plot(U)
        ax1.plot(p_arr[:, 0])
        ax2.plot(p_arr[:, 1])
        plt.show()

    def test_optimize_sinusoidal_waveform_spsa(self):

        mu = 1e-3

        args = [
            '--dt', str(0.01),
            '--N', str(250),
            '--dimensionless_from_physical', str(True),
            '--mu', str(mu),
            '--no-minimize',
            '--normalize_objective',
            '--f_min', str(0.01),
            '--f_max', str(0.1)
            ]

        WFO = WaveformOptimizer(args)
        WFO.set_objective('speed')
        WFO.set_normalizer(linear_norm(0.0, 0.2))
        p0 = [1.0, 1.0]
        history = WFO.optimize_waveform_parameter_spsa(waveform, p0, 0.1, 2, max_iter=50)

        h5_filename = 'sweep_celegans_mu_min=-3.0_mu_max=1.0_mu_step=0.2_c0_min=0.5_c0_max=2.0_c0_step=0.1lam0_min=0.5_lam0_max=2.0_lam0_step=0.1_N=750_dt=0.001.h5'
        h5, PG = load_sweep(test_data_dir / h5_filename)
        c0_arr, lam0_arr = PG.v_from_key('c0'), PG.v_from_key('lam0')
        mu_idx = np.abs(mu - PG.v_from_key('mu')).argmin()
        U = h5['U'][mu_idx, :]

        spline = RectBivariateSpline(lam0_arr, c0_arr, U.T)
        # Create refined grid
        lam0_arr_refine = np.linspace(lam0_arr.min(), lam0_arr.max(), len(lam0_arr) * 100)
        c0_arr_refine = np.linspace(c0_arr.min(), c0_arr.max(), len(c0_arr) * 100)

        # Evaluate the spline on the refined grid
        U_refined = spline(lam0_arr_refine, c0_arr_refine)
        i_max, j_max = np.unravel_index(U_refined.argmax(), U_refined.shape)
        lam0_max, c0_max = lam0_arr_refine[i_max], c0_arr_refine[j_max]

        U_opt_arr = [tup[1] for tup in history]
        p_opt_arr = [tup[0] for tup in history]
        idx = np.argmin(U_opt_arr)
        c0_opt = p_opt_arr[idx][0]
        lam0_opt = p_opt_arr[idx][1]

        print(f'c0_max: brute_force: {c0_max}, optimizer: {c0_opt}')
        print(f'lam0_max: brute_force: {lam0_max}, optimizer: {lam0_opt}')

        self.assertEqual(c0_max, c0_opt)
        self.assertEqual(lam0_max, lam0_opt)


