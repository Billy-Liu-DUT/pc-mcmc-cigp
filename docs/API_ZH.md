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

## `pc_mcmc_cigp.kinetics`

动力学公共库通过 `create_kinetic_template(name)` 创建模板，通过
`TemplateRegistry.describe(name)` 查询输入列、物种和参数边界。目前包括：

| 模板 | 主要用途 | 输入 |
|---|---|---|
| `simple_arrhenius` | 单步双分子反应 | A0, B0, T, t |
| `series_arrhenius` | 连续反应与中间体 | A0, B0, T, t |
| `parallel_arrhenius` | 主副反应竞争 | A0, B0, T, t |
| `reversible_arrhenius` | 可逆反应 | A0, B0, T, t |
| `autocatalytic_arrhenius` | 自催化诱导行为 | A0, P0, T, t |
| `epoxidation` | 环氧化与后续损失 | substrate0, oxidant0, T, t |
| `robertson` | 刚性多时间尺度反应 | A0, B0, C0, t |
| `michaelis_menten` | 饱和酶/催化动力学 | S0, catalyst0, T, t |
| `langmuir_hinshelwood` | 双吸附表面反应 | A0, B0, T, t |
| `power_law` | 经验反应级数 | A0, B0, T, t |
| `inhibited` | 竞争抑制或催化剂抑制 | S0, I0, catalyst, t |
| `radical_chain` | 引发、传播与终止 | substrate0, initiator0, T, t |

这些模板是可组合基础族，不代表穷尽所有命名有机反应。复杂机理应拆成基元步骤，
给每个 `Reaction` 指定质量作用、幂律、Arrhenius、可逆或饱和速率律。

`PathwayGenerator.generate(sources, targets, max_steps)` 从候选反应网枚举最短路径；
将其反应索引传给 `SpikeAndSlabSampler.fit(..., candidate_pathways=...)` 后，采样器会混合
单反应翻转和整条路径翻转，并在 `DiscoveryResult.selected_pathways` 返回后验入选路径。

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
