# `results/` — final evaluation outputs

This directory is **gitignored**. Final paper-ready artifacts go here:

```
results/
├── main_table.csv             # 7-baseline × 4-metric main table
├── pareto_curve.pdf           # EPDMS vs FLOPs Pareto figure
├── ablations/
│   ├── A1_wo_RL.csv
│   ├── ...
│   └── A10_fastv_selector_at_input.csv
└── plots/
    └── *.pdf
```

Source CSVs are the single source of truth; figures are regenerated
from them via `code/scripts/plot_*.py`.
