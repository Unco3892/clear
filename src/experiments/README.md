## COMMANDS TO RUN THE EXPERIMENTS
These set of commands replicate the results for the experiments in the paper on the benchmark datasets.

To run the experiments for the benchmark datasets, use the following command from the current `experiments` directory:

```bash
python benchmark_real_data.py --datasets "ailerons,airfoil,allstate,ca_housing,computer,concrete,diamond,elevator,energy_efficiency,insurance,kin8nm,miami_housing,naval_propulsion,parkinsons,powerplant,qsar,sulfur,superconductor" --coverage 0.99 --generate_tables --n_jobs 24 --global_log
```
You can adjust the `coverage` argument to set the desired coverages to 0.9, 0.95 and 0.99 as per the paper, For example:
```bash
python benchmark_real_data.py --datasets "ailerons,airfoil,allstate,ca_housing,computer,concrete,diamond,elevator,energy_efficiency,insurance,kin8nm,miami_housing,naval_propulsion,parkinsons,powerplant,qsar,sulfur,superconductor" --coverage 0.90 --generate_tables --n_jobs 24 --global_log
```

<!-- Then, combine the results from individual tables into combined ones by running:

```bash
python combine_benchmark_pcs_data.py --coverage 90
python combine_benchmark_pcs_data.py --coverage 95
python combine_benchmark_pcs_data.py --coverage 99
``` -->

Alternatively, you can run everything at once via `run_benchmark_pcs_data.py`:

```bash
python run_benchmark_pcs_data.py
```

In case of any premission problems, run the following before executing `run_benchmark_pcs_data.py`:

```bash
chmod +x src/experiments/run_benchmark_pcs_data.py
```

