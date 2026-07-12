from legacy._kinetics_adapter import LegacyKineticsAdapter
from pc_mcmc_cigp.kinetics import ReversibleArrheniusKinetics

class ReversibleReactionODE(LegacyKineticsAdapter, ReversibleArrheniusKinetics):
    def __init__(self, W_init=None, X_scaler=None, **_kwargs):
        ReversibleArrheniusKinetics.__init__(self); self._legacy_init(W_init, X_scaler)
