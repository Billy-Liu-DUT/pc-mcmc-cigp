from legacy._kinetics_adapter import LegacyKineticsAdapter
from pc_mcmc_cigp.kinetics import SeriesArrheniusKinetics

class SeriesReactionODE(LegacyKineticsAdapter, SeriesArrheniusKinetics):
    def __init__(self, W_init=None, X_scaler=None, **_kwargs):
        SeriesArrheniusKinetics.__init__(self); self._legacy_init(W_init, X_scaler)
