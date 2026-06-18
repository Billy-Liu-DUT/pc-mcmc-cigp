import numpy as np
import GPy
import GPy.util.linalg
from base_physics import BasePhysicsModel  # 从同一目录导入


class CIGP(GPy.core.Model):
    """
    化学信息高斯过程 (CIGP) 模型。
    (这个基类现在是“纯净”的，归一化逻辑将在主脚本中处理)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 physics_model: BasePhysicsModel,
                 kernel: GPy.kern.Kern = None,
                 name: str = 'CIGP'):

        super(CIGP, self).__init__(name=name)
        if not isinstance(physics_model, BasePhysicsModel):
            raise TypeError("physics_model 必须是 BasePhysicsModel 的一个实例。")
        if Y.ndim != 2 or Y.shape[1] != 1:
            raise ValueError(f"Y 必须是 (N, 1) 向量，但得到 {Y.shape}")

        self.X = GPy.core.Param('X', X)
        self.Y = GPy.core.Param('Y', Y)

        self.physics_model = physics_model
        self.W = GPy.core.Param('W', self.physics_model.W)

        if kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True, variance=1.0, lengthscale=1.0)
        else:
            self.kernel = kernel

        self.likelihood = GPy.likelihoods.Gaussian(variance=1.0)

        self.link_parameters(self.W, self.kernel, self.likelihood)

    def parameters_changed(self):
        """
        【已更新】: 这是 CIGP_Normalized *之前*的基类。
        在我们的新策略中，这个基类 *不会* 被直接调用，
        而是由 run_training_example.py 中的“子类” CIGP_Normalized *重写* (override)。
        """
        self.mu_total = self.physics_model.compute_mean(self.X.values, self.W.values)
        self.mu_total = np.nan_to_num(self.mu_total, nan=0.0, posinf=0.0, neginf=0.0)
        self.Y_minus_mu = self.Y - self.mu_total

        # (GPy 内部逻辑...)
        self.K_err = self.kernel.K(self.X)
        self.sigma_n_sq = self.likelihood.variance
        self.Sigma_total = self.K_err + np.eye(self.X.shape[0]) * self.sigma_n_sq
        try:
            self.L_total = np.linalg.cholesky(self.Sigma_total)
            self.woodbury_vector, _ = GPy.util.linalg.dpotrs(self.L_total, self.Y_minus_mu, lower=True)
            self.woodbury_inv, _ = GPy.util.linalg.dpotri(self.L_total, lower=True)
            self.log_det_Sigma = 2.0 * np.sum(np.log(np.diag(self.L_total)))
            self.data_fit = np.dot(self.Y_minus_mu.T, self.woodbury_vector)
        except np.linalg.LinAlgError:
            self.log_det_Sigma = 1e10
            self.data_fit = 1e10

    def log_likelihood(self) -> float:
        nll = 0.5 * self.log_det_Sigma + 0.5 * self.data_fit
        return -nll

    def gradients_W(self) -> np.ndarray:
        dL_dmu = self.woodbury_vector
        dmu_dW = self.physics_model.compute_gradients_W(self.X.values, self.W.values)
        grad_W = np.dot(dL_dmu.T, dmu_dW)
        return grad_W.flatten()

    def _param_grad_helper(self, dL_dK):
        self.W.gradient = self.gradients_W()

    def predict(self, X_new: np.ndarray):
        mu_phys_new = self.physics_model.compute_mean(X_new, self.W.values)
        K_star = self.kernel.K(self.X, X_new)
        mu_err_new = np.dot(K_star.T, self.woodbury_vector)
        mu_pred = mu_phys_new + mu_err_new

        k_star_star_diag = self.kernel.Kdiag(X_new)
        v = GPy.util.linalg.dpotrs(self.L_total, K_star, lower=True)[0]
        v_T_Kinv_v = np.sum(v ** 2, axis=0)
        var_err_pred = k_star_star_diag - v_T_Kinv_v
        var_err_pred = np.maximum(var_err_pred, 1e-9)
        var_pred = var_err_pred + self.likelihood.variance

        return mu_pred, var_pred.reshape(-1, 1)

    # ***************************************************************
    # * 【新功能】: 用于可视化的函数
    # ***************************************************************
    def predict_g_err(self, X_new: np.ndarray):
        """
        只预测 g_err (结构性误差) 的后验均值和方差。
        """
        K_star = self.kernel.K(self.X, X_new)
        mu_err_new = np.dot(K_star.T, self.woodbury_vector)

        k_star_star_diag = self.kernel.Kdiag(X_new)
        v = GPy.util.linalg.dpotrs(self.L_total, K_star, lower=True)[0]
        v_T_Kinv_v = np.sum(v ** 2, axis=0)
        var_err_pred = k_star_star_diag - v_T_Kinv_v
        var_err_pred = np.maximum(var_err_pred, 1e-9)

        return mu_err_new, var_err_pred.reshape(-1, 1)

    def gradients_X(self, dL_dK):
        return np.zeros_like(self.X)

    def gradients_Y(self, dL_dK):
        return np.zeros_like(self.Y)