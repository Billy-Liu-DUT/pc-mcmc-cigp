# Workflow gates

| Stage | Owner | Exit condition |
|---|---|---|
| intake | LLM + user | reviewable project and mode |
| experiment design | LLM proposes, user confirms | executable request matrix |
| data mapping | deterministic parser + user | every required field and unit resolved |
| data validation | deterministic validator | no blocking error |
| mechanism proposal | LLM proposes, compiler checks | balanced finite network |
| mechanism approval | user | explicit approval recorded |
| PC-MCMC | deterministic sampler | diagnostics and posterior saved |
| MCMC review | LLM explains | next action selected |
| CIGP compilation | deterministic compiler | physics model contract saved |
| CIGP fitting | deterministic optimizer | validated fit and uncertainty |
| next experiment | LLM explains, user confirms | new condition request saved |

Only numerical MCMC settings may be adjusted without changing scientific scope. Search-space changes require user confirmation. Optimization-only projects may bypass PC-MCMC by selecting a predefined template. Coupled projects compile the supported network. Iterative projects may return from CIGP to mechanism proposal when new data conflict with the model.
