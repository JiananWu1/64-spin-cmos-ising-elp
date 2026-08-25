# QUBO Energy-Landscape Perturbation Figure Reproduction

This repository contains the problem instances, processed simulation and measurement results, and plotting scripts required to reproduce Figures 6 and 7 for the QUBO energy-landscape perturbation study.

This is a **plotting-only release**. The processed CSV files required by the plotting scripts are included. The hardware-measurement and behavioral-simulation pipelines used to generate these processed results are not included.

## Repository contents

- `data/problems/random_qubo/`: 400 random QUBO instances covering 16, 32, 48, and 64 variables, five edge densities, and 20 instances per size-density configuration.
- `data/problems/seven_queens/`: 7-queens QUBO instances used for the diagonal-to-row/column penalty sweep.
- `results/simulation/random_qubo/`: processed behavioral-simulation data for Figure 6(a-c).
- `results/measurement/random_qubo/`: processed measurement data for Figure 6(d-h).
- `results/measurement/seven_queens/`: processed measurement data and spin assignments for Figure 7.
- `figures/figure_6/figure_6.py`: plotting script for Figure 6.
- `figures/figure_7/figure_7.py`: plotting script for Figure 7.

## Python environment

Python 3.10 or newer is recommended. Install the required packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

Matplotlib is pinned in `requirements.txt` to reduce small version-dependent layout differences. If Arial is installed, the scripts use Arial; otherwise they fall back to Liberation Sans and then DejaVu Sans.

## Reproduce Figure 6

From the repository root, run:

```bash
python figures/figure_6/figure_6.py
```

The script automatically reads the included processed simulation and measurement CSV files and writes:

```text
figures/figure_6/figure_6.pdf
```

## Reproduce Figure 7

From the repository root, run:

```bash
python figures/figure_7/figure_7.py
```

The script automatically reads the included 7-queens QUBO files and processed measurement CSV files and writes:

```text
figures/figure_7/figure_7.pdf
```

## Path handling

All input and output paths are defined relative to the repository location. No file paths need to be supplied on the command line, and the plotting scripts do not depend on the current working directory.