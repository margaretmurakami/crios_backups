# ASTE-Tools

Reusable Python tools for analyzing and visualizing output from the Arctic and
Subpolar gyre sTate Estimate (ASTE).

This directory now contains two layers:

* `aste_tools/`: importable package code for reusable notebook/script tooling.
* `an_helper_functions/`: the original helper files kept for compatibility and
  reference while notebooks are migrated.

## Overview

ASTE-Tools provides a collection of utilities for working with ASTE output, including:

* Loading ASTE diagnostics and state variables
* Computing heat, salt, and freshwater budgets
* Calculating volume, heat, and salt transports through user-defined gates
* Performing analyses in temperature-salinity (T-S) space
* Computing regional budgets and tracer transformations
* Producing publication-quality figures

The package is designed to support reproducible analysis workflows for studies of Arctic Ocean circulation, Atlantification, borealization, freshwater storage, and ocean heat transport.

---

## Features

### Data Access

* Read ASTE NetCDF output
* Read ASTE budget diagnostics
* Load grid geometry and masks
* Handle ASTE LLC grids

### Budget Analysis

Compute contributions from:

* Advection
* Diffusion
* Surface forcing
* Vertical mixing
* Sea ice interactions
* Residual terms

Supported budgets include:

* Heat
* Salt
* Freshwater
* Volume

### Transport Diagnostics

Calculate:

* Volume transport
* Heat transport
* Salt transport

through arbitrary sections and commonly used Arctic gateways:

* Barents Sea Opening (BSO)
* Fram Strait
* Davis Strait
* Bering Strait
* St. Anna Trough
* Other user-defined sections

### T-S Space Diagnostics

* Bin tracers in temperature-salinity space
* Compute heat and salt budgets in T-S coordinates
* Diagnose diapycnal and isopycnal transformations
* Analyze water-mass transformations

### Regional Analysis

Support for:

* Barents Sea
* Eurasian Basin
* Canada Basin
* Arctic Ocean
* Nordic Seas
* User-defined regions

### Visualization

* Maps
* Vertical sections
* Hovmöller diagrams
* T-S diagrams
* Budget closure plots
* Transport time series

---

## Installation

```bash
cd /home/mmurakami/crios_backups/aste_tools
pip install -e .
```

---

## Package Structure

```text
aste_tools/
├── aste_tools/              # installable package
│   ├── grid.py              # ASTE90/ASTE270 specs, compact/tracer/vector transforms
│   ├── binning.py           # T-S binning and G-term accumulation
│   ├── masks.py             # wet-cell indices, area means, 1D/2D to 3D expansion
│   ├── timing.py            # model time-step helpers
│   ├── legacy.py            # loader for original helper modules
│   └── io/binary.py         # big-endian MITgcm binary readers
├── an_helper_functions/     # original notebook helper modules
├── examples/
│   └── demo_synthetic.py    # no-data demo
├── notebooks/
│   └── 00_synthetic_demo.ipynb
└── tests/
```

---

## Quick Start

### Grid Conversion

```python
import numpy as np
from aste_tools.grid import aste270, compact_to_tracer

grid = aste270()
theta_compact = np.zeros(grid.compact_shape_3d)
theta_map = compact_to_tracer(theta_compact, grid)
```

### T-S Binning

```python
from aste_tools.binning import bin_array, create_ts_mesh

salt_bins = bin_array(salinity, salt_edges)
theta_bins = bin_array(theta, theta_edges)
volume_ts = create_ts_mesh(volume, salt_bins, theta_bins, n_salinity=nS, n_theta=nT)
```

### Masked Means

```python
from aste_tools.masks import masked_area_mean

mean_theta, area = masked_area_mean(theta, basin_mask, RAC)
```

### Binary Reads

```python
from aste_tools.io import read_big_endian

RAC = read_big_endian("RAC.data", dtype="float64", shape=grid.compact_shape_2d)
```

### Legacy Notebook Migration

Old notebook cells commonly use:

```python
sys.path.append("/home/mmurakami/crios_backups/an_helper_functions")
from aste_helper_funcs import *
from binning import *
```

Prefer package imports for new work:

```python
from aste_tools.grid import aste90, aste270, compact_to_tracer, tracer_to_compact
from aste_tools.binning import bin_array, create_ts_mesh, calc_g_term
from aste_tools.masks import masked_area_mean, wet_indices_in_mask
from aste_tools.timing import monthly_file_steps, select_month_steps
```

If a notebook still needs a helper that has not been promoted yet:

```python
from aste_tools.legacy import load_legacy_module

old_helpers = load_legacy_module("aste_helper_funcs")
legacy_tracer = old_helpers.get_aste_tracer(field, nfx, nfy)
```

---

## Example Workflow

```text
Load ASTE output
        ↓
Load grid geometry
        ↓
Define region or section
        ↓
Compute budget terms
        ↓
Calculate transports
        ↓
Analyze T-S transformations
        ↓
Generate figures
```

## Useful Existing Notebooks

The notes in `uesful_notebooks.txt` point to these migration targets:

* Gateway validation and gate transports:
  `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/verify_gates_aste90.ipynb`,
  `ASTE_270/Pemberton_BarentsSpaper/Tracking_Atlantification/Arthun_2012_OHT_redone.ipynb`
* Map-view surface and tendency terms:
  `ASTE_270/Pemberton_BarentsSpaper/Tendencies_demo/surface_breakdown.ipynb`
* Full mass, heat, and salt budgets:
  `budgeting/regional_budgeting/fullASTE_budgeting_heat.ipynb`,
  `budgeting/regional_budgeting/fullASTE_budgeting_mass.ipynb`,
  `budgeting/regional_budgeting/fullASTE_budgeting_salt.ipynb`
* Water-mass/T-S transformations:
  `ASTE_270/Pemberton_BarentsSpaper/BarentsS_paper/BarentsS_budget_breakdown.ipynb`,
  `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/ASTE_90_calc/full_wmt_spellout_barentsS_miniaste.ipynb`

Run the no-data demo before connecting to scratch-backed ASTE output:

```bash
cd /home/mmurakami/crios_backups/aste_tools
python examples/demo_synthetic.py
```

---

## Scientific Applications

ASTE-Tools is intended for studies of:

* Arctic Atlantification
* Barents Sea heat accumulation
* Freshwater storage and export
* Ocean heat transport variability
* Water-mass transformation
* Borealization of Arctic gateways
* Arctic climate variability

---

## Testing

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=aste_tools
```

---

## Citation

If you use ASTE-Tools in a publication, please cite:

```text
Author et al. (YEAR)

ASTE-Tools:
A Python package for analysis of Arctic and Subpolar gyre sTate Estimate output.
```

---

## License

MIT License.
