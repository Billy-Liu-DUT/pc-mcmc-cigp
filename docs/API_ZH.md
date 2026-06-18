# API 速查

## `pc_mcmc_cigp.reactions`

### `Species`

定义化学物种。

```python
Species("H2", {"H": 2, "Charge": 0})
```

### `Reaction`

定义候选基元反应。

```python
Reaction([h2], [h_rad, h_rad])
```

常用属性：

- `equation_str`
- `is_balanced(atom_types)`
- `stoichiometric_vector(species_order)`

### `AtomMappedNetworkGenerator`

生成质量/电荷守恒的候选反应。

```python
generator = AtomMappedNetworkGenerator(
    reactants=[h2],
    products=[hbr],
    fragments=[h_rad, br_rad],
)
reactions = generator.generate()
```

## `pc_mcmc_cigp.discovery`

### `MechanismEngine`

质量作用定律 ODE 引擎。

```python
engine = MechanismEngine(species, reactions)
y = engine.simulate(k_vector, z_structure, y0, t_eval)
```

说明：

- 默认优先使用 SciPy `solve_ivp`。
- 没有 SciPy 时回退到 RK4。
- `engine.last_solver_backend` 记录最近一次使用的后端。

### `MCMCConfig`

MCMC 配置对象。

```python
config = MCMCConfig(
    n_steps=10000,
    burn_in=2000,
    prior_sparsity=0.005,
    sigma_likelihood=0.08,
    random_state=42,
)
```

### `SpikeAndSlabSampler`

单链 PC-MCMC。

```python
sampler = SpikeAndSlabSampler(engine, config)
result = sampler.fit(dataset)
```

`dataset` 是字典列表，每个实验包含：

- `t`: 时间点。
- `y0_full`: 全物种初始浓度。
- `data_matrix`: 观测浓度矩阵，shape 为 `(n_observed_species, n_times)`。
- `obs_indices`: 被观测物种在 engine species 中的索引。

### `run_multiple_chains`

多链 PC-MCMC，并返回 R-hat / ESS 诊断。

```python
from pc_mcmc_cigp.discovery import run_multiple_chains

result = run_multiple_chains(engine, dataset, config, n_chains=4)
print(result.diagnostics["rhat_max"])
print(result.diagnostics["ess_min"])
```

### `DiscoveryResult`

主要字段：

- `posterior_inclusion_probabilities`
- `mean_parameters`
- `selected_reactions`
- `map_structure`
- `map_parameters`
- `diagnostics`
- `chain_diagnostics`

## `pc_mcmc_cigp.cigp`

### `CIGPRegressor`

sklearn 风格 CIGP 回归器。

```python
model = CIGPRegressor(physics_model, CIGPConfig()).fit(X, y)
mean, var = model.predict(X_new)
residual_mean, residual_var = model.predict_residual(X_new)
```

`physics_model` 需要提供：

```python
compute_mean(X, W)
compute_gradients_W(X, W)
```

## `pc_mcmc_cigp.acquisition`

### `AcquisitionFactory`

```python
from pc_mcmc_cigp.acquisition import AcquisitionFactory

ei = AcquisitionFactory.create("EI")
gwu = AcquisitionFactory.create("GWU")
dh = AcquisitionFactory.create("DH")
pc_ei = AcquisitionFactory.create("PC_EI")
scores = pc_ei.compute(model, X_candidates, y_best=0.7)
```

可用策略：

- `EI`: Expected Improvement。
- `GWU`: Gradient-Weighted Uncertainty。
- `DH`: Discrepancy Hunter。
- `PC_EI`: Physically Constrained Expected Improvement。
