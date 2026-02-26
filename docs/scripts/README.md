# CLEAR Paper: Table & Plot Generation Scripts

Reproduces all LaTeX tables and figures from the CLEAR paper.

## Prerequisites

```
pip install numpy pandas matplotlib seaborn
```

## Usage

Run from the **project root**:

| Command | What it does |
|---------|--------------|
| `python docs/scripts/generate_tables.py` | Tables only |
| `python docs/scripts/generate_tables.py --plots` | Tables + plots |
| `python docs/scripts/generate_tables.py --plots-only` | Plots only |

On Windows (PowerShell), use backslashes: `python docs\scripts\generate_tables.py --plots`

### Standalone: CLEAR percentage improvement table

```
python docs/scripts/combine_improved_de_sqr_percentage.py \
  --results_dir results/de_sqr --output_dir docs/tex_tbls/de_sqr --coverage 95
```

Options: `--metrics all` for every metric (default: PICP,NIW,NCIW,QuantileLoss), `--save_csv` to also write detail/summary CSVs.

## Output

### Tables (73 files)

```
docs/tex_tbls/
├── table-combined-dataset-stats.tex
├── de_sqr/
│   ├── table-de-sqr-95-{metric}.tex              (8 metrics)
│   ├── table-sqr-de-calibration-95.tex
│   └── table-clear-percentage-improvement-95.tex
└── pcs_cqr/
    ├── standard/
    │   ├── table-clear-vs-uacqr-improvement-95_standard.tex
    │   └── {a,b,c}/
    │       ├── table-combined-95-{metric}-final_standard.tex  (8 metrics)
    │       ├── table-combined-95-gamma-lambda-final_standard.tex
    │       └── table-uncertainty-metrics-95_standard.tex
    └── conformalized/
        ├── table-clear-vs-uacqr-improvement-95_conformalized.tex
        └── {a,b,c}/
            ├── table-combined-95-{metric}-final_conformalized.tex
            ├── table-combined-95-gamma-lambda-final_conformalized.tex
            └── table-uncertainty-metrics-95_conformalized.tex
```

Metrics (8): PICP, NIW, MPIW, QuantileLoss, CRPS, NCIW, ExpectileLoss, IntervalScoreLoss.

Variants: **(a)** qPCS\_all\_10seeds\_all, **(b)** qPCS\_qxgb\_10seeds\_qxgb, **(c)** PCS\_all\_10seeds\_qrf.

The UACQR improvement tables compare CLEAR variant (c) against UACQR-S/UACQR-P. The **standard** version uses uncalibrated CLEAR (main paper), the **conformalized** version uses CLEAR-c (appendix).

### Plots

All plots are saved directly to `docs/figures/` mirroring `paper/overleaf/figures/`:

```
docs/figures/pcs_cqr/standard/real/         # Real benchmark bar plots
docs/figures/pcs_cqr/standard/simulations/  # Simulation distance metric plots
docs/figures/uacqr/                         # UACQR comparison plots
```

## Scripts

| Script | Role |
|--------|------|
| `generate_tables.py` | Main entry point -- tables + plot orchestration |
| `utils.py` | Shared helpers (`format_metric_name`, `write_latex_table`) |
| `combine_real_benchmark_results.py` | PCS/CQR metric table aggregation |
| `combine_improved_de_sqr.py` | DE/SQR metric table aggregation |
| `plot_real_benchmark_results.py` | Real benchmark bar plots |
| `plot_simulation_results.py` | Simulation distance metric plots |
| `generate_uacqr_csv_plots.py` | UACQR comparison plots |
| `combine_improved_de_sqr_percentage.py` | CLEAR % improvement over DE/SQR (Table \ref{tab:clear_percentage_improvement}) |

## Data Dependencies

- `results/standard/` -- Standard PCS/CQR benchmark CSVs (variants a, b, c)
- `results/conformalized/` -- Conformalized benchmark CSVs
- `results/de_sqr/` -- DE/SQR comparison CSVs
- `results/uacqr/` -- UACQR aggregated results (standard and conformalized CSVs, summary)
- `docs/figures/pcs_cqr/standard/simulations/` -- Simulation CSV data (coverage + width results)
- `data/` -- Raw dataset CSVs (for dataset statistics table)
