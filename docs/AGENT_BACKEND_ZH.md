# 反应动力学 Agent 后端

该后端把自然语言协调和确定性算法分开：LLM 负责澄清目标、设计实验、提出候选机理并解释结果；CSV 解析、单位转换、守恒检查、PC-MCMC、ODE 和 CIGP 由本地代码执行。模型不得虚构实验或算法结果。

## 工作模式

- `mechanism_only`：只完成机理空间设计、PC-MCMC 和路径解释。
- `optimization_only`：已有可信动力学模型，直接选择公共 CIGP 模板。
- `coupled`：PC-MCMC 筛选机理后，将后验网络编译成 CIGP 物理均值。
- `iterative_coupled`：CIGP 新实验可以更新参数，也可以在证据冲突时返回机理发现。

## 状态机

主流程为：

```text
intake
  -> experiment_plan_ready
  -> waiting_for_data
  -> data_mapping
  -> data_validation
  -> mechanism_proposal
  -> waiting_for_mechanism_approval
  -> mcmc_running
  -> mcmc_review
  -> cigp_model_compilation
  -> cigp_fitting
  -> optimization_ready
  -> waiting_for_next_experiment
```

仅优化模式可从数据验证直接进入 `cigp_model_compilation`。机理模式可在 MCMC 审查后结束。迭代模式在新数据否定当前模型时返回 `mechanism_proposal`。

## CSV 数据契约

上传格式沿用仓库 benchmark：一行代表一个实验、重复和时间点。

```csv
experiment_id,replicate,time_s,temperature_K,A0_mol_L,A_mol_L,P_mol_L,yield
E001,1,0,323.15,1.0,1.0,0.0,0.0
E001,1,60,323.15,1.0,0.72,0.25,0.25
```

`BenchmarkDataMapper` 支持明确的列别名、分钟转秒、摄氏度转开尔文以及百分数产率转小数。模糊单位必须由用户确认。原始 CSV、映射报告、标准行和最终算法输入分别保存，不静默覆盖。

标准行可以编译成：

- PC-MCMC：`t`、`y0_full`、`data_matrix`、`obs_indices`，可附带 `temperature` 和实验条件；
- CIGP：按模板 `input_names` 排序的 `X` 与目标 `y`。

## PC-MCMC 与 CIGP 对接

PC-MCMC 输出反应和路径包含概率、参数区间、R-hat、ESS、无效求解次数和后验预测误差。改变候选中间体、基元反应、先验、可逆性或参数边界前必须得到用户确认；链数、步数、预热和 proposal scale 等数值配置可以依据诊断调整。

`CompiledNetworkKinetics` 将通过包含概率阈值的反应步骤转成：

```text
dC/dt = S r(C, T, theta)
```

化学计量矩阵来自已编译机理，速率律来自每个基元步骤，PC-MCMC 参数区间中点作为 CIGP 初值。CIGP 仍学习物理模型不能解释的残差。预定义模板路线保持可用，因此机理发现和优化既能耦合，也能分开运行。

## 专有 Skill 与 Schema

运行时包含七个专业 Skill：

- `reaction_intake`
- `experiment_design`
- `data_template`
- `data_quality`
- `mechanism_hypothesis`
- `mcmc_interpretation`
- `cigp_optimization`

每个 Skill 有独立 Prompt 和严格 JSON Schema。结构化结果不合格时只允许进行一次无新增科学主张的格式修复；再次失败则停止。对话元数据记录 `active_skill` 和 `schema_repaired`。`GET /api/skills` 返回可供前端调试的阶段、说明和 Schema。

离线检查：

```powershell
E:\Anaconda3\envs\bayesian\python.exe scripts\evaluate_agent_prompts.py
```

配置兼容 API 后可以串行实测：

```powershell
E:\Anaconda3\envs\bayesian\python.exe scripts\evaluate_agent_prompts.py --live
```

## 启动工作台

在设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL` 的同一终端中运行：

```powershell
E:\Anaconda3\envs\bayesian\python.exe scripts\run_agent_web.py
```

访问 `http://127.0.0.1:8765`。外部模型请求通过全局信号量强制单并发。不要启动多个共享同一密钥的服务进程，因为进程间不能共享该锁。

## 当前仍需完善

- 长链 PC-MCMC 的后台任务队列、取消、进度推送和断点恢复；
- GC/HPLC 厂商文件及 Excel 多表自动适配；
- 前端完整的字段映射确认、机理差异审批和动态 CIGP 模型展示；
- 更大规模真实有机反应数据上的 Prompt 与机理候选生成评测。
