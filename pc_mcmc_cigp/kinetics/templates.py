from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pc_mcmc_cigp.kinetics.core import KineticModel, KineticParameter, SimulationResult, arrhenius, integrate_ode


P_ORDER = (0.0, 4.0)
P_LOG_A = (-6.0, 18.0)
P_LOG_EA = (0.0, 6.0)


class ODEKineticModel(KineticModel):
    target_index = -1

    def initial_state(self, row: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rhs(self, row: np.ndarray, W: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
        raise NotImplementedError

    def simulate(self, X, W=None) -> SimulationResult:
        Xp = self.validate_X(X)
        W = self.W if W is None else np.asarray(W, dtype=float)
        self.validate_parameters(W)
        values, trajectories, solvers = [], [], []
        for row in Xp:
            state, solver = integrate_ode(self.rhs(row, W), self.initial_state(row), float(row[-1]))
            trajectories.append(state)
            values.append(state[self.target_index])
            solvers.append(solver)
        return SimulationResult(np.asarray(values).reshape(-1, 1), tuple(trajectories), True, "+".join(sorted(set(solvers))))


class SimpleArrheniusKinetics(ODEKineticModel):
    input_names = ("A0", "B0", "temperature", "time")
    species_names = ("A", "B", "P")
    target_index = 2

    def __init__(self, input_transform=None):
        super().__init__([
            KineticParameter("order_A", 1.0, P_ORDER), KineticParameter("order_B", 1.0, P_ORDER),
            KineticParameter("log10_A", 5.0, P_LOG_A), KineticParameter("log10_Ea", 4.5, P_LOG_EA),
        ], input_transform)

    def initial_state(self, row): return np.array([row[0], row[1], 0.0])
    def rhs(self, row, W):
        k = arrhenius(W[2], W[3], row[2])
        return lambda _t, y: np.array([-1, -1, 1]) * k * max(y[0], 0)**W[0] * max(y[1], 0)**W[1]


class SeriesArrheniusKinetics(ODEKineticModel):
    input_names = ("A0", "B0", "temperature", "time")
    species_names = ("A", "B", "I", "P")
    target_index = 3
    def __init__(self, input_transform=None):
        super().__init__([
            KineticParameter("order_A_1", 1., P_ORDER), KineticParameter("order_B_1", 1., P_ORDER),
            KineticParameter("log10_A_1", 5., P_LOG_A), KineticParameter("log10_Ea_1", 4.5, P_LOG_EA),
            KineticParameter("order_I_2", 1., P_ORDER), KineticParameter("log10_A_2", 5., P_LOG_A),
            KineticParameter("log10_Ea_2", 4.7, P_LOG_EA),
        ], input_transform)
    def initial_state(self, row): return np.array([row[0], row[1], 0., 0.])
    def rhs(self, row, W):
        k1, k2 = arrhenius(W[2], W[3], row[2]), arrhenius(W[5], W[6], row[2])
        def f(_t, y):
            r1 = k1 * max(y[0], 0)**W[0] * max(y[1], 0)**W[1]; r2 = k2 * max(y[2], 0)**W[4]
            return np.array([-r1, -r1, r1-r2, r2])
        return f


class ParallelArrheniusKinetics(ODEKineticModel):
    input_names = ("A0", "B0", "temperature", "time")
    species_names = ("A", "B", "P", "S")
    target_index = 2
    def __init__(self, input_transform=None):
        params=[]
        for suffix, la, le in (("main", 5., 4.5), ("side", 4., 4.8)):
            params += [KineticParameter(f"order_A_{suffix}",1.,P_ORDER), KineticParameter(f"order_B_{suffix}",1.,P_ORDER), KineticParameter(f"log10_A_{suffix}",la,P_LOG_A), KineticParameter(f"log10_Ea_{suffix}",le,P_LOG_EA)]
        super().__init__(params,input_transform)
    def initial_state(self,row): return np.array([row[0],row[1],0.,0.])
    def rhs(self,row,W):
        k1,k2=arrhenius(W[2],W[3],row[2]),arrhenius(W[6],W[7],row[2])
        def f(_t,y):
            r1=k1*max(y[0],0)**W[0]*max(y[1],0)**W[1]; r2=k2*max(y[0],0)**W[4]*max(y[1],0)**W[5]
            return np.array([-r1-r2,-r1-r2,r1,r2])
        return f


class ReversibleArrheniusKinetics(ODEKineticModel):
    input_names=("A0","B0","temperature","time"); species_names=("A","B","P"); target_index=2
    def __init__(self,input_transform=None):
        super().__init__([KineticParameter("order_A",1.,P_ORDER),KineticParameter("order_B",1.,P_ORDER),KineticParameter("order_P",1.,P_ORDER),KineticParameter("log10_A_fwd",5.,P_LOG_A),KineticParameter("log10_Ea_fwd",4.5,P_LOG_EA),KineticParameter("log10_A_rev",3.,P_LOG_A),KineticParameter("log10_Ea_rev",4.8,P_LOG_EA)],input_transform)
    def initial_state(self,row): return np.array([row[0],row[1],0.])
    def rhs(self,row,W):
        kf,kr=arrhenius(W[3],W[4],row[2]),arrhenius(W[5],W[6],row[2])
        return lambda _t,y: np.array([-1,-1,1])*(kf*max(y[0],0)**W[0]*max(y[1],0)**W[1]-kr*max(y[2],0)**W[2])


class AutocatalyticArrheniusKinetics(ODEKineticModel):
    input_names=("A0","P0","temperature","time"); species_names=("A","P"); target_index=1
    def __init__(self,input_transform=None): super().__init__([KineticParameter("order_A",1.,P_ORDER),KineticParameter("order_P",1.,P_ORDER),KineticParameter("log10_A",5.,P_LOG_A),KineticParameter("log10_Ea",4.5,P_LOG_EA)],input_transform)
    def initial_state(self,row): return np.array([row[0],row[1]])
    def rhs(self,row,W):
        k=arrhenius(W[2],W[3],row[2]); return lambda _t,y: np.array([-1,1])*k*max(y[0],0)**W[0]*max(y[1],0)**W[1]


class EpoxidationKinetics(ODEKineticModel):
    input_names=("styrene0","peracid0","temperature","time"); species_names=("styrene","peracid","epoxide","acid"); target_index=2
    def __init__(self,input_transform=None): super().__init__([KineticParameter("log10_A_1",6.,P_LOG_A),KineticParameter("log10_Ea_1",4.7,P_LOG_EA),KineticParameter("log10_A_2",10.,P_LOG_A),KineticParameter("log10_Ea_2",4.9,P_LOG_EA)],input_transform)
    def initial_state(self,row): return np.array([row[0],row[1],0.,0.])
    def rhs(self,row,W):
        k1,k2=arrhenius(W[0],W[1],row[2]),arrhenius(W[2],W[3],row[2])
        def f(_t,y):
            r1=k1*max(y[0],0)*max(y[1],0); r2=k2*max(y[2],0)*max(y[3],0); return np.array([-r1,-r1,r1-r2,r1-r2])
        return f


class RobertsonKinetics(ODEKineticModel):
    input_names=("A0","B0","C0","time"); species_names=("A","B","C"); target_index=2
    def __init__(self,input_transform=None): super().__init__([KineticParameter("log10_k1",-1.398,(-12,12)),KineticParameter("log10_k2",4.778,(-12,12)),KineticParameter("log10_k3",7.477,(-12,12))],input_transform)
    def initial_state(self,row): return row[:3].copy()
    def rhs(self,row,W):
        k1,k2,k3=10**W
        def f(_t,y):
            a,b,c=np.maximum(y,0); r1=k1*a; r2=k2*b*b; r3=k3*b*c; return np.array([-r1+r3,r1-r2-r3,r2])
        return f


class MichaelisMentenKinetics(ODEKineticModel):
    input_names=("substrate0","enzyme0","temperature","time"); species_names=("S","P"); target_index=1
    def __init__(self,input_transform=None): super().__init__([KineticParameter("log10_kcat",1.,(-8,12)),KineticParameter("log10_Km",-1.,(-12,6)),KineticParameter("log10_Ea",4.,P_LOG_EA)],input_transform)
    def initial_state(self,row): return np.array([row[0],0.])
    def rhs(self,row,W):
        vmax=arrhenius(W[0],W[2],row[2])*row[1]; km=10**W[1]
        return lambda _t,y: np.array([-1,1])*vmax*max(y[0],0)/(km+max(y[0],0)+1e-15)


class LangmuirHinshelwoodKinetics(ODEKineticModel):
    input_names=("A0","B0","temperature","time"); species_names=("A","B","P"); target_index=2
    def __init__(self,input_transform=None): super().__init__([KineticParameter("log10_A",5.,P_LOG_A),KineticParameter("log10_Ea",4.5,P_LOG_EA),KineticParameter("log10_K_A",0.,(-12,12)),KineticParameter("log10_K_B",0.,(-12,12))],input_transform)
    def initial_state(self,row): return np.array([row[0],row[1],0.])
    def rhs(self,row,W):
        k=arrhenius(W[0],W[1],row[2]); ka,kb=10**W[2],10**W[3]
        def f(_t,y):
            a,b=np.maximum(y[:2],0); r=k*ka*a*kb*b/(1+ka*a+kb*b)**2; return np.array([-r,-r,r])
        return f


class PowerLawKinetics(ODEKineticModel):
    input_names=("A0","B0","temperature","time"); species_names=("A","B","P"); target_index=2
    def __init__(self,input_transform=None): super().__init__([KineticParameter("k",1.,(1e-12,1e6)),KineticParameter("order_A",1.,P_ORDER),KineticParameter("order_B",1.,P_ORDER)],input_transform)
    def initial_state(self,row): return np.array([row[0],row[1],0.])
    def rhs(self,row,W): return lambda _t,y: np.array([-1,-1,1])*W[0]*max(y[0],0)**W[1]*max(y[1],0)**W[2]


class InhibitedKinetics(ODEKineticModel):
    input_names=("S0","I0","catalyst","time"); species_names=("S","P"); target_index=1
    def __init__(self,input_transform=None): super().__init__([KineticParameter("kcat",1.,(1e-12,1e6)),KineticParameter("Km",1.,(1e-12,1e6)),KineticParameter("Ki",1.,(1e-12,1e6))],input_transform)
    def initial_state(self,row): return np.array([row[0],0.])
    def rhs(self,row,W): return lambda _t,y: np.array([-1,1])*W[0]*row[2]*max(y[0],0)/(W[1]*(1+row[1]/W[2])+max(y[0],0)+1e-15)


class RadicalChainKinetics(ODEKineticModel):
    input_names=("substrate0","initiator0","temperature","time"); species_names=("S","R","P"); target_index=2
    def __init__(self,input_transform=None): super().__init__([KineticParameter("log10_A_init",3.,P_LOG_A),KineticParameter("log10_Ea_init",4.,P_LOG_EA),KineticParameter("log10_A_prop",5.,P_LOG_A),KineticParameter("log10_Ea_prop",4.,P_LOG_EA),KineticParameter("k_term",1.,(1e-12,1e8))],input_transform)
    def initial_state(self,row): return np.array([row[0],0.,0.])
    def rhs(self,row,W):
        ki,kp=arrhenius(W[0],W[1],row[2]),arrhenius(W[2],W[3],row[2])
        def f(_t,y):
            s,r,_=np.maximum(y,0); init=ki*row[1]; prop=kp*s*r; term=W[4]*r*r; return np.array([-prop,init-term,prop])
        return f


class TemplateRegistry:
    _templates: dict[str,type[KineticModel]]={}
    @classmethod
    def register(cls,name):
        def deco(model): cls._templates[name]=model; return model
        return deco
    @classmethod
    def create(cls,name,**kwargs):
        if name not in cls._templates: raise KeyError(f"unknown kinetic template {name!r}; available: {', '.join(cls.names())}")
        return cls._templates[name](**kwargs)
    @classmethod
    def names(cls): return tuple(sorted(cls._templates))
    @classmethod
    def describe(cls,name):
        model=cls.create(name); return {"name":name,"inputs":model.input_names,"species":model.species_names,"parameters":[p.__dict__ for p in model.parameters]}


for _name,_class in {
    "simple_arrhenius":SimpleArrheniusKinetics,"series_arrhenius":SeriesArrheniusKinetics,"parallel_arrhenius":ParallelArrheniusKinetics,
    "reversible_arrhenius":ReversibleArrheniusKinetics,"autocatalytic_arrhenius":AutocatalyticArrheniusKinetics,"epoxidation":EpoxidationKinetics,
    "robertson":RobertsonKinetics,"michaelis_menten":MichaelisMentenKinetics,"langmuir_hinshelwood":LangmuirHinshelwoodKinetics,
    "power_law":PowerLawKinetics,"inhibited":InhibitedKinetics,"radical_chain":RadicalChainKinetics,
}.items(): TemplateRegistry._templates[_name]=_class

def create_kinetic_template(name:str,**kwargs)->KineticModel: return TemplateRegistry.create(name,**kwargs)
def list_kinetic_templates()->tuple[str,...]: return TemplateRegistry.names()
