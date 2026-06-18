# -*- coding: utf-8 -*-
import numpy as np
import GPy
from scipy.stats import qmc
import config
from cigp_model_normalized import CIGP_Normalized
from acquisition_functions import AcquisitionFactory
# 导入所有 5 个算子
from simple_reaction import SimpleReactionODE
from reversible_reaction import ReversibleReactionODE
from parallel_reaction import ParallelReactionODE
from series_reaction import SeriesReactionODE
from autocatalytic_reaction import AutocatalyticODE


def get_lhs_starting_points(n_points: int, bounds_dict: dict):
    phys_params = config.CURRENT_PARAM_NAMES
    gp_params = ['k_var', 'l_1', 'l_2', 'l_3', 'l_4', 'n_var']
    full_param_names = phys_params + gp_params

    l_bounds = [bounds_dict.get(n, config.SAMPLING_BOUNDS.get(n))[0] for n in full_param_names]
    u_bounds = [bounds_dict.get(n, config.SAMPLING_BOUNDS.get(n))[1] for n in full_param_names]

    sampler = qmc.LatinHypercube(d=len(full_param_names))
    return qmc.scale(sampler.random(n=n_points), l_bounds, u_bounds)


def run_optimization_task(task_id: int, start_point_ND: np.ndarray,
                          X_norm: np.ndarray, Y_norm: np.ndarray,
                          w_bounds: dict, X_scaler: dict, Y_scaler: dict):
    try:
        if not X_norm.flags['F_CONTIGUOUS']: X_norm = np.asfortranarray(X_norm)
        if not Y_norm.flags['F_CONTIGUOUS']: Y_norm = np.asfortranarray(Y_norm)

        n_phys = len(config.CURRENT_PARAM_NAMES)
        w_start = start_point_ND[0:n_phys]

        # GP Params
        k_var, l1, l2, l3, l4, n_var = start_point_ND[n_phys:n_phys + 6]
        l_scales = np.array([l1, l2, l3, l4])

        # --- 算子工厂 ---
        model_type = config.PHYSICS_MODEL_TYPE

        if model_type == 'SIMPLE':
            physics_model = SimpleReactionODE(w_start, X_scaler)

        elif 'REVERSIBLE' in model_type:
            is_var = ('VAR' in model_type)
            physics_model = ReversibleReactionODE(w_start, X_scaler, variable_orders=is_var)

        elif 'PARALLEL' in model_type:
            physics_model = ParallelReactionODE(w_start, X_scaler, variable_orders=True)

        elif 'SERIES' in model_type:
            physics_model = SeriesReactionODE(w_start, X_scaler)

        elif 'AUTOCAT' in model_type:
            physics_model = AutocatalyticODE(w_start, X_scaler)

        else:
            raise ValueError(f"Unknown Model: {model_type}")

        # --- 构建模型 ---
        kernel = GPy.kern.RBF(input_dim=X_norm.shape[1], ARD=True, variance=k_var, lengthscale=l_scales)
        model = CIGP_Normalized(X_norm, Y_norm, physics_model, Y_scaler, kernel=kernel,
                                lambda_penalty=config.PENALTY_LAMBDA)
        model.likelihood.variance = n_var

        # --- 约束 ---
        for i, name in enumerate(config.CURRENT_PARAM_NAMES):
            model.W[i:i + 1].constrain_bounded(w_bounds[name][0], w_bounds[name][1])

        model.likelihood.variance.constrain_bounded(config.CONSTRAINT_NOISE_VAR[0], config.CONSTRAINT_NOISE_VAR[1])
        model.kernel.variance.constrain_bounded(config.CONSTRAINT_KERNEL_VAR[0], config.CONSTRAINT_KERNEL_VAR[1])
        model.kernel.lengthscale.constrain_positive(config.CONSTRAINT_KERNEL_LEN)

        model.optimize(config.OPTIMIZER, messages=False, max_iters=config.MAX_ITERS)

        return (model.objective_function(), model.W.values.copy(), model.param_array.copy())

    except Exception as e:
        return (np.inf, start_point_ND[0:n_phys], None)