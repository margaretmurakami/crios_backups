# crios_backups

`crios_backups` is a research-code archive for PhD oceanography work centered on ASTE, Barents Sea water-mass transformation, regional heat/salt/mass budgets, WAOM float analysis, ECCO/Lab Sea tutorials, and related proposal/manuscript material.

The repository is intentionally archival: it preserves active notebooks, helper code, generated figures, validation experiments, and writing drafts. The main organizing document is [DEMO_SHEET.md](DEMO_SHEET.md), which provides a categorized guide to the folders and recommended entry points.

## Repository Structure

| Path | Description |
| --- | --- |
| `ASTE_270/` | Main ASTE270 analysis archive, including Barents Sea diagnostics, mapping, layer/WMT validation, surface-forcing breakdowns, figure workflows, and manuscript-oriented notebooks. |
| `ASTE_90/` | ASTE90/miniASTE diagnostics and cleaned-up layer examples used for smaller validation runs. |
| `budgeting/` | Regional heat, salt, and mass budget notebooks for full ASTE, ASTE90, and Lab Sea configurations, plus shared run configuration code. |
| `an_helper_functions/` | Reusable Python helpers for ASTE grids, binary reads, T-S binning, J terms, vector handling, plotting, and budget calculations. |
| `matlab/` | MATLAB utilities and a rewritten demo script for converting compact/global ASTE tracer and vector fields into a plotted ASTE domain. |
| `WAOM/` | WAOM float modification, water-mass identification, clustering, figure scripts, transects, Argo comparison, and polynya filtering workflows. |
| `Extended_labsea_tutorial/` | Lab Sea tutorial notebooks for reading data, checking T-S diagnostics, state verification, and volumetric distributions. |
| `ECCO_hackathon/` | ECCO/PODAAC tutorial and ECCO volume-budget closure example notebooks. |
| `overleaf/` | Proposal and manuscript source material, including figures, templates, BibTeX files, and compiled documents. |

## Main Scientific Threads

The most developed thread is the Barents Sea budget and water-mass transformation workflow. It spans physical-space budgets, T-S/layer-space budgets, gateway and surface terms, G/J/M-term diagnostics, single-cell validation, ASTE90/miniASTE tests, ASTE270 validation, and manuscript-style figure production.

Other major threads include WAOM float analysis, Lab Sea and ECCO tutorial material, reusable ASTE helper functions, and Overleaf proposal/manuscript documents.

## Recommended Entry Points

- Broad archive guide: [DEMO_SHEET.md](DEMO_SHEET.md)
- Current miniASTE/layer validation: `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/FULL_MINIASTE_WALKTHROUGH.ipynb`
- ASTE270 all-equations validation: `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_aste270/validate_alleqs_aste270.ipynb`
- Regional budget setup: `budgeting/regional_budgeting/FULLASTE_MULTI_RUN_SETUP.md`
- Shared Python helpers: `an_helper_functions/`
- MATLAB ASTE-domain demo: `matlab/example_get_vector.m`
- WAOM float workflow: `WAOM/Modifying_Floats/ID_WaterMasses.ipynb` and `WAOM/figure_scripts/create_all_plots.ipynb`

## Notes

Many notebooks depend on local model output, grid files, binary MITgcm diagnostics, or generated intermediate products that are not necessarily stored in this repository. Treat paths in notebooks and MATLAB scripts as examples to update for the machine and run configuration being used.

The working tree may contain generated notebook outputs, image products, checkpoints, and cache files. For a formal demo or handoff, start from the documents above and run only the notebooks/scripts relevant to the target workflow.
