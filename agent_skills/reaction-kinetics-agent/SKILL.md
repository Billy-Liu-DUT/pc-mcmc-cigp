---
name: reaction-kinetics-agent
description: Coordinate organic-reaction kinetic projects from natural-language intake and benchmark-compatible CSV data through experiment design, PC-MCMC mechanism discovery, mechanism-to-CIGP model compilation, sequential optimization, and result interpretation. Use when an agent must track whether work is in data collection, mechanism discovery, or CIGP optimization; propose user-confirmed mechanism-space changes; map uploaded kinetic data; or recommend the next experiment without inventing algorithm results.
---

# Reaction Kinetics Agent

Run a gated, auditable experiment-mechanism-optimization loop. Treat the LLM as coordinator and hypothesis generator; call deterministic code for data, chemistry, sampling, fitting, and diagnostics.

## Workflow

1. Structure known facts, hypotheses, objective, reactor, species, conditions, observables, and constraints.
2. Select `mechanism_only`, `optimization_only`, `coupled`, or `iterative_coupled`.
3. Design time-resolved discriminating experiments for PC-MCMC or objective-focused experiments for CIGP.
4. Map uploaded CSV columns into the benchmark-compatible contract. Require confirmation for ambiguous fields or units.
5. Validate values, coverage, replicates, conditions, conservation when available, and identifiability.
6. Propose a finite candidate species and elementary-step network. Compile it and request user approval before PC-MCMC.
7. Tune numerical MCMC settings from diagnostics. Request approval before changing intermediates, reactions, priors, reversibility, or parameter bounds.
8. Interpret posterior support as evidence, never proof. Gate coupled optimization on convergence or an explicit override.
9. Compile the supported mechanism into a CIGP physics model, or select a predefined template for optimization-only work.
10. Recommend the next experiment with a declared purpose: mechanism discrimination, parameter identification, optimization, or validation.
11. Return to mechanism discovery if new data contradict the compiled model.

## Guardrails

- Preserve raw uploads and record every mapping and transformation.
- Never claim an algorithm ran unless a tool result is present.
- Never silently remove data or alter scientific search-space boundaries.
- Keep units explicit and keep user-visible explanations separate from structured artifacts.
- Serialize provider calls at concurrency one.

Read [references/workflow.md](references/workflow.md) for gates and ownership. Read [references/data-contract.md](references/data-contract.md) when handling CSV files. Read [references/scientific-guardrails.md](references/scientific-guardrails.md) before mechanism or posterior interpretation.
