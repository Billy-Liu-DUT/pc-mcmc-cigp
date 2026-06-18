# 文章定位与投稿策略

文章题目：

> Synergizing Physically Constrained MCMC and Chemical-Informed Gaussian Processes for Reaction Network Discovery

## 文章当前定位

这篇文章更适合放在以下交叉方向：

- computational chemical engineering
- process systems engineering
- digital chemistry
- interpretable machine learning for reaction kinetics
- Bayesian optimization for chemical processes

它不太像传统有机化学或单纯计算化学文章。核心贡献应该讲成：

1. 物理约束的 Spike-and-Slab MCMC 用于反应网络拓扑发现。
2. Chemical-Informed GP 用于机理参数估计、结构残差建模和主动学习。
3. HBr 自由基链反应与 styrene epoxidation 优化作为两个 benchmark。

## 投稿优先级

### 1. Computers & Chemical Engineering

推荐优先投。

理由：

- 范围非常匹配：建模、数值分析、优化、智能系统、process systems engineering。
- Elsevier 官方页面显示它支持 open access，但也保留 subscription 路径；非 OA 通常不收 APC。
- 对“算法 + 化工过程应用 + 优化”的稿件更友好。

需要注意：

- 叙事应从 chemistry discovery 稍微转向 process systems engineering。
- 标题可以考虑更工程化，例如：

```text
Physically Constrained Bayesian Learning for Reaction Network Discovery and Process Optimization
```

官方页面：

- https://www.sciencedirect.com/journal/computers-and-chemical-engineering/publish/guide-for-authors

### 2. Digital Discovery

主题很契合，但费用风险更高。

理由：

- RSC 官方页面写明它是面向 machine learning、AI、automation 和 digital transformation of chemistry 的期刊。
- Scope 明确包括 Bayesian optimization、design of experiments、interpretable models。

风险：

- RSC 页面显示它是 Gold Open Access，APC 可能适用；如果学校有 RSC open access agreement，可能免 APC。

官方页面：

- https://www.rsc.org/publishing/journals/digital-discovery

### 3. Chemical Science

高风险冲刺选项。

理由：

- RSC 旗舰，跨化学科学，影响力更高。
- 如果能把“发现隐藏自由基链机理”和“打破纯数据驱动黑箱”的化学意义讲得非常强，可以尝试。

风险：

- 门槛明显更高。
- RSC 官方页面显示：accepted articles submitted on or after 1 July 2026 will be subject to an APC。
- 如果想避免 APC，需要确认投稿/接收时间和学校协议。

官方页面：

- https://www.rsc.org/publishing/journals/chemical-science

### 4. Journal of Cheminformatics

方向部分匹配，但不符合“最好免费”的目标。

理由：

- 适合 cheminformatics、molecular modelling 和开源算法。

风险：

- Springer Nature 官方页面显示 APC 为 £1690 / $2390 / €1990。
- 文章需要更偏 cheminformatics，当前更像反应动力学/过程系统工程。

官方页面：

- https://link.springer.com/journal/13321/how-to-publish-with-us#Fees%20and%20funding

## 我的建议

第一目标：`Computers & Chemical Engineering`。

备选冲刺：如果愿意承担 APC 或学校能免 APC，可以考虑 `Digital Discovery`。

不建议优先：`Journal of Cheminformatics`，除非后续把稿件改成更强的软件/cheminformatics 工具论文。

## 投稿前必须修改

1. 删除重复 Abstract。当前 docx 有两个 Abstract。
2. 明确说明实验是 synthetic benchmark / simulated benchmark，不要写得像真实湿实验。
3. 增加 Code Availability，引用本仓库和 `experiments/` 复现脚本。
4. 降低 AI 论文腔：减少被动语态和模板化大词。
5. 强化消融实验：分别展示无 sparsity prior、无 thermodynamic constraints、纯 SINDy、纯 GP。
6. 在 Discussion 里承认限制：候选反应池依赖先验、MCMC 成本高、真实实验噪声和未观测中间体会更难。
7. 如果投 Computers & Chemical Engineering，把 Fig.4 的 process optimization 和 safety/regret 讲得更重。
