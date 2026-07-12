from legacy._kinetics_adapter import LegacyKineticsAdapter
from pc_mcmc_cigp.kinetics import ParallelArrheniusKinetics

class ParallelReactionODE(LegacyKineticsAdapter, ParallelArrheniusKinetics):
    def __init__(self, W_init=None, X_scaler=None, **_kwargs):
        ParallelArrheniusKinetics.__init__(self); self._legacy_init(W_init, X_scaler)
