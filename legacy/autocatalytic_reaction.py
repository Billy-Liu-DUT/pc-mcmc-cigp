from legacy._kinetics_adapter import LegacyKineticsAdapter
from pc_mcmc_cigp.kinetics import AutocatalyticArrheniusKinetics

class AutocatalyticODE(LegacyKineticsAdapter, AutocatalyticArrheniusKinetics):
    def __init__(self, W_init=None, X_scaler=None, **_kwargs):
        AutocatalyticArrheniusKinetics.__init__(self); self._legacy_init(W_init, X_scaler)
