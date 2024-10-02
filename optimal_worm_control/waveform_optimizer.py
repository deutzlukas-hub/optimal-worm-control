# From built-in
from multiprocessing import Pool
from typing import List, Optional, Callable
from os import cpu_count
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
# From third-party
import numpy as np
# From worm-rod engine
from worm_rod_engine.worm import Worm
# From optimal-worm-control
from optimal_worm_control.util import parameters_from_args

class WaveformOptimizer():

    def __init__(self, args = [], quiet=False):

        self.params = parameters_from_args(args)
        self._init_logger(quiet)
        self._init_control()

    def _init_logger(self, quiet: bool):

        self.logger = logging.getLogger(__name__)

        if quiet:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.INFO)
        # Add a console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Optional: Add a formatter for better log message formatting
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        # Add the handler to the logger
        self.logger.addHandler(console_handler)

    def _init_worm(self):

        return Worm(2,
            self.params['numerical_param'],
            self.params['dimensionless_param'],
            self.params['output_param']
        )

    def _init_control(self):

        dt, N = self.params['numerical_param'].dt, self.params['numerical_param'].N
        T_sim = self.params['control_param'].T_sim
        self.t_arr = np.arange(dt, T_sim + 0.1 * dt, dt)
        self.s_arr = np.linspace(0.0, 1.0, N)

        if self.params['control_param'].gsmo:
            s0_h = self.params['control_param'].s0_h
            Ds_h = self.params['control_param'].Ds_h
            s0_t = self.params['control_param'].s0_t
            Ds_t = self.params['control_param'].Ds_t

            self.sig_h = 1.0 / (1 + np.exp(-(self.s_arr - s0_h) / Ds_h))
            self.sig_t = 1.0 / (1 + np.exp(+(self.s_arr - s0_t) / Ds_t))

        if self.params['control_param'].fmts:
            t_on = self.params['control_param'].t_on
            tau_on = self.params['control_param'].tau_on
            self.sig_on = 1.0 / (1 + np.exp(-(t_on - tau_on) / t_on))


    def _generate_input_curvature(self, p_arr: List[float]):

        k0_arr = np.zeros((len(self.t_arr), len(self.s_arr)))

        for i, t in enumerate(self.t_arr):
            s_arr = (self.s_arr - p_arr[-1]*t) % 1.0
            k0 = self.waveform(s_arr, p_arr)
            if self.params['control_param'].gsmo:
                k0 = self.sig_h * self.sig_t * k0
            if self.params['control_param'].fmts:
                k0 = self.sig_on * k0
            k0_arr[i, :] = k0

        return k0_arr

    def _solve(self, p_arr: List[float]):

        k0 = self._generate_input_curvature(p_arr)
        worm = self._init_worm()
        output, e = worm.solve(self.params['control_param'].T_sim, k0=k0, progress=False)
        assert output['exit_status']
        return output

    def _speed(self, p_arr: List[float] | np.ndarray):

        output = self._solve(p_arr)
        t = output['FS'].t
        idx_arr = t >= (self.params['control_param'].T_sim - 1)
        r = output['FS'].r[idx_arr, :]
        R = r.mean(axis=-1)
        U = np.linalg.norm(R[-1] - R[0])
        return U

    def _compute_objective(self, p_arr: List[float]):
        obj = self.objective_function(p_arr)
        if self.params['optimizer_param'].normalize:
            obj = self.norm(obj)
        return obj

    def _spsa_perturbation(self, params: np.ndarray, epsilon: float, perturbation: np.ndarray):
        """
        Function to compute the objective at the perturbed points.

        :param objective_function: The objective function to evaluate.
        :param params: The current parameter vector.
        :param epsilon: The perturbation magnitude.
        :param perturbation: The random perturbation vector.
        :return: Difference in objective function values for the perturbed parameters.
        """
        # Perturbed parameters
        params_pos = params + epsilon * perturbation
        params_neg = params - epsilon * perturbation

        # Evaluate objective at perturbed points
        f_pos = self._compute_objective(params_pos)
        f_neg = self._compute_objective(params_neg)

        return f_pos, f_neg, perturbation

    def _numerical_gradient(self,
        params: List[float],
        epsilon: float,
        K: int,
        workers: Optional[int] = None):
        # Use parallel finite differences or SPSA for gradient approximation

        num_params = len(params)
        gradient_estimate = np.zeros(num_params)

        if workers is None:
            workers = K if cpu_count() >= K else cpu_count()

        # # Run the perturbations in parallel
        # with ThreadPoolExecutor(max_workers=workers) as executor:
        #     futures = []
        #
        #     for _ in range(K):
        #         # Generate random perturbation vector (e.g., uniform between -1 and 1)
        #         perturbation = np.random.uniform(-1, 1, size=params.shape)
        #         # Submit the SPSA perturbation task to the executor for parallel computation
        #         futures.append(executor.submit(self._spsa_perturbation, params, epsilon, perturbation))

            # Create a pool of workers
        with Pool(processes=workers) as pool:
            futures = []
            for _ in range(K):
                # Generate random perturbation vector (e.g., uniform between -1 and 1)
                perturbation = np.random.uniform(-1, 1, size=params.shape)
                # Submit the SPSA perturbation task to the pool for parallel computation
                futures.append(pool.apply_async(self._spsa_perturbation, (params, epsilon, perturbation)))

            # Collect the results from the parallel computations
            #for future in as_completed(futures):
            for future in futures:
                f_pos, f_neg, perturbation = future.get()
                #f_pos, f_neg, perturbation = future.result()
                # Update the gradient estimate for this perturbation
                gradient_estimate += (f_pos - f_neg) / (2 * epsilon * perturbation)




        # Average the gradient estimates over K perturbations
        gradient_estimate /= K
        return gradient_estimate

    #================================================================================================
    # Class API
    #================================================================================================

    def set_normalizer(self, norm: callable):
        self.norm = norm

    def set_objective(self, objective_function: str | Callable):

        if isinstance(objective_function, str):
            if objective_function == 'speed':
                self.objective_function = self._speed
        else:
            self.objective_function = objective_function

    def  optimize_waveform_parameter_spsa(self,
        waveform: callable,
        p0: List[float],
        epsilon: float,
        K: int,
        max_iter: int = 100,
        workers: Optional[int] = None,
        tol = 1e-4):

        self.waveform = waveform

        p_arr = np.array(p0)
        obj_value = self._compute_objective(p_arr)
        history = [(p_arr, obj_value)]

        self.logger.info(f'Initial p_arr: {p_arr}')
        self.logger.info(f'Initial objective-value: {obj_value}')

        for iteration in range(max_iter):
            self.logger.info(f"Iteration {iteration}:")
            # Compute the numerical gradient
            gradient = self._numerical_gradient(p_arr, epsilon, K, workers)
            self.logger.info(f'Gradient: {gradient}')
            # Update the parameters: gradient descent step
            if self.params['optimizer_param'].minimize:
                p_arr -= epsilon * gradient  # Adjust the update step as needed
            else:
                p_arr += epsilon * gradient

            self.logger.info(f'p_arr: {p_arr}')
            # Evaluate the objective function with the updated parameters
            obj_value = self._compute_objective(p_arr)
            self.logger.info(f'Objective-value: {obj_value}')
            # Store the history of parameter values and objective values
            history.append((p_arr.copy(), obj_value))

            # Check for convergence (if needed, implement a convergence criterion)
            if np.linalg.norm(gradient) < tol:  # Example convergence criterion
                print(f"Converged after {iteration} iterations.")
                break

        return history

if __name__ == '__main__':
    pass



















