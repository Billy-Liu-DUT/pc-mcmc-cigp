# Data Extracted From Source Materials

This directory stores small, curated CSV files used by figure-generation
scripts.

Current files:

- `fig4_cigp.csv`: CIGP optimization log extracted from `图改.pptx`.
- `fig4_standard_bo.csv`: Standard BO baseline log extracted from `图改.pptx`.
- `fig4_cigp_vs_standard_bo.png`: generated comparison figure.

Regenerate from the source PPTX:

```powershell
E:\Anaconda3\envs\bayesian\python.exe scripts\extract_ppt_optimization_logs.py "D:\工作二连续流\图\图改.pptx" data
E:\Anaconda3\envs\bayesian\python.exe scripts\plot_fig4.py
```
