from __future__ import annotations

import warnings

import numpy as np

from pc_mcmc_cigp.kinetics import InputTransform


class LegacyKineticsAdapter:
    def _legacy_init(self, W_init=None, X_scaler=None) -> None:
        warnings.warn(
            f"{type(self).__name__} is deprecated; import its replacement from pc_mcmc_cigp.kinetics",
            DeprecationWarning, stacklevel=3,
        )
        if X_scaler is not None:
            self.input_transform = InputTransform.from_legacy(X_scaler)
        if W_init is not None:
            self.W = np.asarray(W_init, dtype=float)
