# ASTE-Tools

Python tools for analyzing and visualizing output from the Arctic and Subpolar gyre sTate Estimate (ASTE).

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

Clone the repository:

```bash
git clone https://github.com/your-org/aste-tools.git
cd aste-tools
```

Install:

```bash
pip install -e .
```

---

## Package Structure

```text
aste_tools/
├── aste_tools/
│   ├── io/
│   ├── grid/
│   ├── budgets/
│   ├── transports/
│   ├── ts_space/
│   ├── regions/
│   ├── plotting/
│   └── utils/
│
├── notebooks/
│   ├── 01_load_data.ipynb
│   ├── 02_heat_budget.ipynb
│   ├── 03_salt_budget.ipynb
│   ├── 04_transport_analysis.ipynb
│   └── 05_ts_space_analysis.ipynb
│
├── tests/
├── docs/
└── README.md
```

---

## Quick Start

### Load ASTE Output

```python
from aste_tools.io import open_aste

ds = open_aste("state.nc")
```

### Compute Heat Budget

```python
from aste_tools.budgets import heat_budget

budget = heat_budget(ds)
```

### Compute Gate Transport

```python
from aste_tools.transports import heat_transport

oht = heat_transport(
    ds,
    gate="BSO"
)
```

### Plot Results

```python
from aste_tools.plotting import plot_timeseries

plot_timeseries(oht)
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
