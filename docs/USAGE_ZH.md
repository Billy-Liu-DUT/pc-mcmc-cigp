# PC-MCMC-CIGP 使用说明

这个项目现在整理成了一个可安装的 Python 包：`pc_mcmc_cigp`。它包含两条主线：

1. `PC-MCMC`：从候选基元反应中推断哪些反应真实存在。
2. `CIGP`：把物理/机理 ODE 作为 GP 的均值函数，用 GP 建模残差，并用于主动学习优化。

旧脚本仍保留在根目录，主要作为历史参考。新的推荐入口是 `pc_mcmc_cigp/`、`examples/` 和 `experiments/`。

## 运行环境

PyCharm 配置里记录的解释器是：

```powershell
E:\Anaconda3\envs\bayesian\python.exe
```

推荐从项目根目录运行：

```powershell
cd D:\工作二连续流\CIGP
```

## 快速示例

先确认环境可用：

```powershell
E:\Anaconda3\envs\bayesian\python.exe scripts\run_smoke_tests.py
```

机理发现：

```powershell
E:\Anaconda3\envs\bayesian\python.exe examples\discover_hbr_mechanism.py
```

输出每条候选反应的 PIP，即 posterior inclusion probability。PIP 越高，MCMC 越认为该反应应当属于真实机理。

CIGP 主动优化：

```powershell
E:\Anaconda3\envs\bayesian\python.exe examples\optimize_epoxidation_cigp.py
```

输出每轮实验条件、观测产率和 best-so-far 产率。

## 论文复现实验

Fig.3 类机理发现数据：

```powershell
E:\Anaconda3\envs\bayesian\python.exe experiments\fig3_hbr_discovery.py
```

输出：

```text
examples/outputs/fig3_hbr/summary.json
```

Fig.4 类主动优化数据：

```powershell
E:\Anaconda3\envs\bayesian\python.exe experiments\fig4_epoxidation_bo.py
```

输出：

```text
examples/outputs/fig4_epoxidation/optimization_history.csv
```

`examples/outputs/` 中的结果文件默认被 `.gitignore` 忽略。复现结果时重新运行脚本即可。

生成图像：

```powershell
E:\Anaconda3\envs\bayesian\python.exe scripts\plot_fig3.py
E:\Anaconda3\envs\bayesian\python.exe scripts\plot_fig4.py
```

如果要使用 `D:\工作二连续流\图\图改.pptx` 里的原始优化日志生成 CIGP vs Standard BO 对比图：

```powershell
E:\Anaconda3\envs\bayesian\python.exe scripts\extract_ppt_optimization_logs.py "D:\工作二连续流\图\图改.pptx" data
E:\Anaconda3\envs\bayesian\python.exe scripts\plot_fig4.py
```

输出：

```text
data/fig4_cigp.csv
data/fig4_standard_bo.csv
data/fig4_cigp_vs_standard_bo.png
```

旧研究脚本已经集中到 `legacy/`。新用户或投稿复现应优先使用 `pc_mcmc_cigp/`、`examples/`、`experiments/` 和 `scripts/`。

## 核心 API

### 反应网络

```python
from pc_mcmc_cigp.reactions import Species, Reaction, AtomMappedNetworkGenerator
```

- `Species`：定义物种名称和原子/电荷组成。
- `Reaction`：定义基元反应。
- `AtomMappedNetworkGenerator.generate()`：生成守恒的候选反应网络。

### PC-MCMC

```python
from pc_mcmc_cigp.discovery import MechanismEngine, MCMCConfig, SpikeAndSlabSampler
```

- `MechanismEngine.simulate(...)`：用质量作用定律积分 ODE。
- `SpikeAndSlabSampler.fit(dataset)`：返回 `DiscoveryResult`。
- `DiscoveryResult.posterior_inclusion_probabilities`：每条候选反应的 PIP。
- `DiscoveryResult.map_structure`：后验中最优结构。

多链诊断：

```python
from pc_mcmc_cigp.discovery import run_multiple_chains

result = run_multiple_chains(engine, dataset, config, n_chains=4)
print(result.diagnostics["rhat_max"])
print(result.diagnostics["ess_min"])
```

### CIGP

```python
from pc_mcmc_cigp.cigp import CIGPRegressor, CIGPConfig
```

`CIGPRegressor` 是 sklearn 风格接口：

```python
model = CIGPRegressor(physics_model, CIGPConfig()).fit(X, y)
mean, var = model.predict(X_new)
residual_mean, residual_var = model.predict_residual(X_new)
```

其中 `physics_model` 需要实现：

```python
compute_mean(X, W)
compute_gradients_W(X, W)
```

## 当前算法还可以继续加强的点

- MCMC：增加多链采样、R-hat、ESS、自适应 proposal step size。
- ODE：进一步暴露 `LSODA`、`BDF`、`Radau` 参数；对失败积分返回更细诊断。
- CIGP：升级到 GPyTorch，支持 ARD、多输出浓度曲线和更大的数据集。
- 热力学约束：把 `mu`、`G`、`Ea`、`k_forward/k_reverse` 做成独立物理约束模块。
- 复现：增加固定图表脚本，把 JSON/CSV 进一步渲染成论文 Fig.3/Fig.4。
