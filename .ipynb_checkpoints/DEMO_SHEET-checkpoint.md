# crios_backups Demo Sheet

This repository is a working archive of PhD analysis code: mostly Jupyter notebooks, helper functions, figures, and writing drafts around ASTE, Barents Sea water-mass transformation, regional budgets, WAOM floats, and related oceanographic diagnostics.

## Quick Inventory

- Main analysis stack: Python/Jupyter, with some MATLAB utilities and Fortran reference/source snippets.
- Tracked contents: roughly 480 notebooks, 50+ Python files, hundreds of generated figures, and Overleaf manuscript/proposal material.
- Main model/data contexts: ASTE270, ASTE90/miniASTE, Lab Sea tutorials, ECCO examples, and WAOM float analysis.
- Most developed scientific thread: Barents Sea heat/salt/volume budgets in physical space and T-S/layer space, including gateways, surface forcing, WMT, and validation of closed budgets.

## Categories

### Mapping and Map Setup

Use these for cartopy setup, ASTE grid geometry, bathymetry, masks, gateways, and paper/pre-quals maps.

- `ASTE_270/mapping/` - early ASTE270 mapping tests, cartopy setup, longitude/latitude extraction, and pre-quals map creation.
- `ASTE_270/mapping/create_map_FORPREQUALS.ipynb` - polished map workflow for proposal/pre-quals material.
- `ASTE_270/Pemberton_BarentsSpaper/BarentsS_paper/create_maps.ipynb` - Barents Sea paper map generation.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/mapping.ipynb` - current Barents Sea mapping notebook.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/miniaste/mapping_miniaste.ipynb` - miniASTE map setup.
- `ASTE_270/Pemberton_BarentsSpaper/BarentsS_paper_old/mapping_xyz_analysis.ipynb` and `mapping_anomalies_depth.ipynb` - older spatial anomaly/depth mapping work.

### Plotting Trends, Time Series, and Hovmollers

Use these for trend lines, seasonal cycles, multi-year comparisons, and time-varying budget terms.

- `ASTE_270/offline_binning/sample_images/basic_intro_trendlines.png` - example trendline output.
- `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/trends_maps.ipynb` - trend maps for the Barents Sea story.
- `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/timeseries_hovmoller.ipynb` - time-series and Hovmoller diagnostics.
- `ASTE_270/Pemberton_BarentsSpaper/BarentsS_paper/time_series_Jterms_heat.ipynb` - heat J-term time series.
- `ASTE_270/Pemberton_BarentsSpaper/BarentsS_paper/time_series_Jterms_salt.ipynb` - salt J-term time series.
- `ASTE_270/Pemberton_BarentsSpaper/BarentsS_paper/time_series_lotsofterms_singlept.ipynb` - many-term single-point budget time series.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/years_comparison.ipynb` - interannual comparison workflow.
- `ASTE_270/Pemberton_BarentsSpaper/presentation_fall_2025/compare_2007_2016.ipynb` and `2007vs2016_forpresentation.ipynb` - presentation-oriented year comparisons.

### Plotting Transport and Gateways

Use these for Atlantic Water transport, gateway fluxes, boundary exchanges, and transects.

- `ASTE_270/Pemberton_BarentsSpaper/BarentsS_paper/AW_transport.ipynb` - Atlantic Water transport analysis.
- `ASTE_270/Pemberton_BarentsSpaper/FullRundown_2years_3_26/gateway_transport.ipynb` - gateway transport calculations for a two-year rundown.
- `ASTE_270/Pemberton_BarentsSpaper/Masking_discussion_april2/gateway_transport.ipynb` - transport work tied to masking choices.
- `ASTE_270/Pemberton_BarentsSpaper/Pemberton2015/Figure11_boundaryexchanges.ipynb` and `BarentsS_boundaryexchanges_fig11.ipynb` - Pemberton-style boundary exchange figures.
- `Extended_labsea_tutorial/uv_transp_Tbin.png` and `temp_sv_transp.png` - Lab Sea transport examples.
- `WAOM/figure_scripts/Figure9_transects.ipynb` - WAOM float/transect plotting.

### Plotting WMT, T-S Budgets, and Layer Budgets

Use these for water-mass transformation, T-S distributions, layer budgets, G/J/M terms, and closure checks.

- `ASTE_270/Pemberton_BarentsSpaper/Pemberton2015/Figure11_12_TS_WMT.ipynb` - WMT figure replication/reference.
- `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/wmt_spellout_bigaste.ipynb` - full ASTE WMT spellout.
- `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/ASTE_90_calc/wmt_budget_spellout.ipynb` - ASTE90 WMT budget walkthrough.
- `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/ASTE_90_calc/wmt_budget_spelloutsaltGOOD.ipynb` - salt WMT version marked as good.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/FULL_MINIASTE_WALKTHROUGH.ipynb` - miniASTE full layer/WMT validation path.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/validate_eq1_2_3.ipynb`, `validate_eq4.ipynb`, `validate_eq5.ipynb` - equation-by-equation validation.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_aste270/validate_alleqs_aste270.ipynb` - ASTE270 all-equations validation.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/check_adv_closure.ipynb` and `adv_closure_TS.py` - advective closure tests in T-S/layer space.
- `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/Mterm_rabbithole/` - detailed M-term debugging notebooks.
- `ASTE_270/Pemberton_BarentsSpaper/Layers/` - layer definitions, budget notebooks, and saved `G_data.pkl`.

### Regional Heat, Salt, and Mass Budgets

Use these for budget equation implementations, ASTE/Lab Sea compatibility, and reusable run configuration.

- `budgeting/regional_budgeting/fullASTE_budgeting_heat.ipynb`, `fullASTE_budgeting_salt.ipynb`, `fullASTE_budgeting_mass.ipynb` - full ASTE regional budget notebooks.
- `budgeting/regional_budgeting/fullASTE_budgeting_heat_aste90.ipynb`, `fullASTE_budgeting_salt_aste90.ipynb`, `fullASTE_budgeting_mass_aste90.ipynb` - ASTE90 variants.
- `budgeting/regional_budgeting/fullASTE_budgeting_heat_labsea.ipynb`, `fullASTE_budgeting_salt_labsea.ipynb`, `fullASTE_budgeting_mass_labsea.ipynb` - Lab Sea variants.
- `budgeting/regional_budgeting/fullaste_run_config.py` - shared setup/configuration helper for ASTE270, ASTE90, and Lab Sea runs.
- `budgeting/regional_budgeting/FULLASTE_MULTI_RUN_SETUP.md` - notes on how to use the shared run configuration.
- `budgeting/labsea/` - one-face Lab Sea heat/salt/mass budget tests and equation verification.
- `ASTE_90/lookat_layers_cleanedup.ipynb` and `ASTE_90/TS_diagnostics_miniaste.py` - ASTE90/minimal ASTE diagnostics.

### Surface Forcing, Sea Ice, and Flux Breakdowns

Use these for surface heat/salt forcing, sea-ice salt contribution, SFLUX terms, and air-sea flux decomposition.

- `ASTE_270/Pemberton_BarentsSpaper/Tendencies_demo/surface_forcing_tendencies.ipynb` - early surface forcing tendency diagnostics.
- `ASTE_270/Pemberton_BarentsSpaper/Tendencies_demo/surface_breakdown.ipynb` - surface term breakdown.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/surface_term_breakdown.ipynb` - miniASTE surface term breakdown.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/surface_breakdown_cleanup.ipynb` and `surface_breakdown_cleanup_salt.ipynb` - cleaned heat/salt surface breakdowns.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_aste270/surface_term_breakdown_SFLUX270.ipynb` - ASTE270 SFLUX breakdown.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/labsea/labsea_surfacebreakdown.ipynb` - Lab Sea surface flux breakdown.
- `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/labsea/*.F` - Fortran references for sea-ice growth, bulk formulae, oceanic surface fluxes, and salt plume forcing.
- `ASTE_270/offline_binning/sample_images/` - sea-ice salt contribution examples, including annual, March, and September salt-ice-component plots.
- `ASTE_270/Pemberton_BarentsSpaper/ASTE_figs/OHC_SI_OHT.ipynb` - ocean heat content, sea ice, and ocean heat transport figure workflow.

### WAOM Floats, Water Masses, and Figures

Use these for WAOM float releases, clustering/categorization, water-mass IDs, seasonality, Argo comparison, and figure scripts.

- `WAOM/Modifying_Floats/categorize.py` and `categorize.sh` - float categorization driver.
- `WAOM/Modifying_Floats/clustering.ipynb` - clustering workflow.
- `WAOM/Modifying_Floats/ID_WaterMasses.ipynb` - water-mass identification.
- `WAOM/Modifying_Floats/Table2_KMeans_grouping.ipynb` - KMeans grouping table workflow.
- `WAOM/Modifying_Floats/combine_floats_subsect.ipynb` and `make_fullnc_file.ipynb` - float file combination and NetCDF creation.
- `WAOM/figure_scripts/create_all_plots.ipynb` - plot-generation hub.
- `WAOM/figure_scripts/Figure2_seasonality.ipynb`, `Figure4_lat_densgrad.ipynb`, `fig2_4_5.ipynb`, `Figure9_transects.ipynb` - paper-style figure notebooks.
- `WAOM/figure_scripts/compare_to_argo.ipynb` - Argo comparison.
- `WAOM/figure_scripts/filter_polynyas_060225.ipynb` - polynya filtering workflow.

### Colormaps, Plotting Helpers, and Reusable Utilities

Use these when starting new notebooks or cleaning older ones.

- `ASTE_270/colormaps/test_colors.ipynb` - colormap experiments.
- `ASTE_270/colormaps/test_isopycnal_mixing.ipynb` - isopycnal mixing color/diagnostic exploration.
- `an_helper_functions/plotting_helpers.py` - reusable plotting helpers.
- `an_helper_functions/binning.py` - T-S/binning utilities.
- `an_helper_functions/aste_helper_funcs.py` - ASTE helper functions.
- `an_helper_functions/get_Jterms.py` and `get_Jterms_gates.py` - J-term calculations.
- `an_helper_functions/prep_grid.py` and `prep_grid_aste_90.py` - grid preparation.
- `an_helper_functions/create_budgO.py`, `calc_UV_conv_1face.py`, `calc_mskmean_T_mod.py`, `mk3D_mod.py` - budget/math utilities.
- `matlab/` - MATLAB versions/examples for reading ASTE tracers/vectors.

### Literature Replication and Scientific Context

Use these for comparison to published Barents Sea/Atlantification diagnostics and manuscript figures.

- `ASTE_270/Pemberton_BarentsSpaper/Pemberton2015/` - replications/adaptations of Pemberton 2015 figures: T-S diagrams, WMT, salt budget, cross-sections, and boundary exchanges.
- `ASTE_270/Pemberton_BarentsSpaper/Lind2018/` - Lind 2018 replication, polar-front analysis, profiles, cross-sections, and time series.
- `ASTE_270/Pemberton_BarentsSpaper/Tracking_Atlantification/` - Atlantification story notebooks and Arthun-style OHT work.
- `ASTE_270/Pemberton_BarentsSpaper/FullRundown_2years_3_26/Barton_2018_fig2.ipynb` - Barton 2018-style diagnostic.
- `overleaf/Quals_proposal_MURAKAMI/` - quals proposal source, figures, and PDF.
- `overleaf/BarentsS_PROPOSAL_v01_26/` - Barents Sea proposal/manuscript material.

### Tutorials and External Data Examples

Use these as examples for reading data, checking T-S diagnostics, and ECCO/PODAAC workflows.

- `Extended_labsea_tutorial/` - Lab Sea reading, T-S diagnostics, state verification, and volumetric distribution tutorial notebooks.
- `ECCO_hackathon/Tutorial_Python3_Jupyter_Notebook_Downloading_ECCO_Datasets_from_PODAAC.ipynb` - PODAAC/ECCO data download tutorial.
- `ECCO_hackathon/ecco_volbudget_closure.ipynb` - ECCO volume budget closure.

## Year-by-Year Rundown

### 2023

- Started the backup repository in December 2023.
- The repo begins as a personal archive for Jupyter notebooks and related PhD work.

### 2024

- Built the foundation for ASTE/Barents Sea analysis: grid handling, cartopy mapping, basin masks, gateway definitions, and reading monthly budget/state output.
- Developed T-S binning and volumetric distributions, then extended that into heat/salt tendency and budget calculations.
- Worked through physical-space and T-S/layer-space budget mechanics: advective fluxes, surface forcing, G/J terms, residuals, and single-cell/small-patch validation.
- Began reproducing and adapting Pemberton-style and Lind-style Barents Sea figures.
- Added ECCO examples and started Overleaf proposal/manuscript material.

### 2025

- Turned the 2024 mechanics into more complete Barents Sea story notebooks: trends, Hovmollers, polar-front masks, MLD, gateway transports, Atlantic Water transport, and 2007/2014/2016 comparisons.
- Dug deeply into WMT, G/J/M terms, and ASTE90/ASTE270 validation, including many focused M-term debugging notebooks.
- Built presentation and proposal material, including fall 2025 presentation notebooks and continued Overleaf edits.
- Added WAOM float analysis: categorization, KMeans grouping, water-mass IDs, polynya filtering, seasonality, transects, and Argo comparison.
- Expanded manuscript-style figure production for Barents Sea budgets, cross-sections, transports, and Atlantification.

### 2026

- Consolidated the layer/WMT budget work into more formal validation workflows for miniASTE, ASTE90, ASTE270, and Lab Sea.
- Added equation-by-equation validation notebooks, full miniASTE walkthroughs, gate-vs-surface comparisons, and advective closure scripts.
- Developed surface flux and SFLUX breakdown workflows for miniASTE, ASTE270, and Lab Sea, including Fortran references for sea-ice growth, oceanic surface fluxes, salt plume forcing, and bulk formulae.
- Added shared configuration for full ASTE multi-run regional budgeting so heat, salt, and mass budgets can be run across ASTE270, ASTE90, and Lab Sea with less notebook-specific setup.
- Continued current Barents Sea figure/diagnostic work around climatology, seasonal cycles, fullyear G plots, and cross-model closed mass budgets.

## Good Entry Points

- For a broad Barents Sea narrative: start with `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/README.txt`, then `trends_maps.ipynb`, `timeseries_hovmoller.ipynb`, and `wmt_spellout_bigaste.ipynb`.
- For current validation work: start with `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/FULL_MINIASTE_WALKTHROUGH.ipynb`.
- For regional budgets across model configurations: start with `budgeting/regional_budgeting/FULLASTE_MULTI_RUN_SETUP.md`.
- For WAOM floats: start with `WAOM/Modifying_Floats/ID_WaterMasses.ipynb` and `WAOM/figure_scripts/create_all_plots.ipynb`.
- For reusable code: start with `an_helper_functions/` and `budgeting/regional_budgeting/fullaste_run_config.py`.
