"""Open-source implementation of the PC-MCMC-CIGP reaction discovery framework."""

from pc_mcmc_cigp.cigp import CIGPConfig, CIGPRegressor
from pc_mcmc_cigp.discovery import DiscoveryResult, MCMCConfig, MechanismEngine, SpikeAndSlabSampler
from pc_mcmc_cigp.reactions import AtomMappedNetworkGenerator, Reaction, Species

__version__ = "0.1.0"

__all__ = [
    "AtomMappedNetworkGenerator",
    "CIGPConfig",
    "CIGPRegressor",
    "DiscoveryResult",
    "MCMCConfig",
    "MechanismEngine",
    "Reaction",
    "Species",
    "SpikeAndSlabSampler",
    "__version__",
]
