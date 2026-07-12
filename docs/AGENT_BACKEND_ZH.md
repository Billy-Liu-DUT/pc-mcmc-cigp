# 反应动力学 Agent 后端

该模块先实现不依赖 LLM API 的确定性工作流。LLM 将来只负责把自然语言转成这里定义的结构化协议，并调用经批准的工具；ODE、PC-MCMC 和 CIGP 仍由本地算法执行。

## 当前流程

1. `ReactionProjectSpec` 保存反应目标、物种、变量、观测量和缺失信息。
2. `ReactionAgentWorkflow` 依据状态机推进项目，不允许跳过数据验证或机理批准。
3. `ExperimentPlanner` 生成多温度、时间序列和初始浓度扰动实验。
4. `ExperimentDataValidator` 检查 CSV 并输出可辨识性风险。
5. `MechanismCompiler` 把结构化基元步骤编译为化学计量矩阵、速率律和 ODE 引擎。
6. `AlgorithmService` 调用多链 PC-MCMC并输出面向 Agent 的摘要。
7. `CIGPService` 选择兼容模板并在收敛门控后推荐实验条件。

## 项目目录

每个项目包含：

```text
project.json
events.jsonl
datasets/
experiment_requests/
mechanisms/
mcmc_runs/
cigp_runs/
reports/
```

所有数据、机理和结果按 `v001`、`v002` 递增保存，不覆盖历史版本。

## 为可视化前端准备的接口

`FrontendReadModel` 已提供：

- 项目阶段和当前允许操作；
- 各类版本化产物数量；
- 候选机理的节点和边；
- 六个页面的稳定名称；
- OpenAI API 是否配置。

建议后续前端页面：

1. 项目向导；
2. 实验计划与 CSV 下载；
3. 数据质量和可辨识性；
4. 候选机理网络；
5. MCMC PIP、R-hat、ESS 和后验预测；
6. CIGP 推荐条件、置信区间和实验历史。

正式前端开发前，应在这些读模型之上增加 FastAPI 或等价 HTTP 层，不要让浏览器直接访问算法对象。

## OpenAI API 占位

`config/agent_runtime.example.json` 默认：

```json
{"api_enabled": false, "openai_model": null}
```

当前不需要 API Key。以后接入时复制为本地配置，并通过环境变量 `OPENAI_API_KEY` 提供密钥。模型名称保持配置化，不写死在算法代码中。

## 尚未实现

- OpenAI Responses API / Agents SDK；
- 自然语言到 `ReactionProjectSpec` 和 `MechanismSpec` 的转换；
- HTTP 服务和可视化网页；
- 仪器厂商原始格式适配；
- 后台任务队列与长时间 MCMC 进度推送。
