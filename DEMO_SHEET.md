# crios_backups Demo Sheet

This repository is a working archive of PhD analysis code: mostly Jupyter notebooks, helper functions, figures, and writing drafts around ASTE, Barents Sea water-mass transformation, regional budgets, WAOM floats, and related oceanographic diagnostics.

## Quick Inventory

- Main analysis stack: Python/Jupyter, with some MATLAB utilities and Fortran reference/source snippets.
- Tracked contents: roughly 480 notebooks, 50+ Python files, hundreds of generated figures, and Overleaf manuscript/proposal material.
- Main model/data contexts: ASTE270, ASTE90/miniASTE, Lab Sea tutorials, ECCO examples, and WAOM float analysis.
- Most developed scientific thread: Barents Sea heat/salt/volume budgets in physical space and T-S/layer space, including gateways, surface forcing, WMT, and validation of closed budgets.

## Formal Demo Path

Use this path when presenting the repository as a coherent research archive rather than as a file dump.

1. Start with the top-level [README.md](README.md) to explain the purpose of the archive and the major folders.
2. Use this demo sheet to choose a scientific thread: Barents Sea WMT/budgets, regional heat/salt/mass budgets, WAOM floats, Lab Sea/ECCO tutorials, or manuscript/proposal material.
3. For the main Barents Sea thread, open `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/FULL_MINIASTE_WALKTHROUGH.ipynb` to show the formal miniASTE/layer validation workflow.
4. Move to `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_aste270/validate_alleqs_aste270.ipynb` for the ASTE270-scale validation.
5. Use `budgeting/regional_budgeting/FULLASTE_MULTI_RUN_SETUP.md` to show how regional budget notebooks are organized across ASTE270, ASTE90, and Lab Sea runs.
6. Use `matlab/example_get_vector.m` to demonstrate the rewritten MATLAB entry point for converting compact/global ASTE vector fields into a plotted ASTE-domain layout.

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

### MATLAB ASTE-Domain Utilities

Use these when a MATLAB workflow needs to reshape compact/global ASTE fields into a map-like ASTE domain.

- `matlab/example_get_vector.m` - rewritten demo script with a configuration block, file checks, vector conversion, grid-mask conversion, plotting, and optional binary export.
- `matlab/get_aste_vector.m` - converts compact/global/cell-array U and V fields into the stitched ASTE domain, including vector sign and C-grid shifts.
- `matlab/get_aste_tracer.m` - converts compact/global/cell-array/gcmfaces tracer fields into the stitched ASTE domain.
- `matlab/sym_g_mod.m` - symmetry/rotation helper used by the ASTE tracer and vector reshaping functions.

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

### 2023 -- working with the budgets and modifying MATLAB code to Python

- Started the backup repository in December 2023 as a working archive for PhD analysis code, notebooks, helper scripts, and model-output experiments.
- The earliest technical thread is budget closure in MATLAB: loading ASTE fields, reshaping compact/global output into the ASTE domain, and checking that the volume/tracer budget terms can be read, transformed, and compared consistently.
- Established the first practical "load layers, close budgets" workflow: read layer and budget diagnostics, convert vector/tracer fields into usable map layouts, and verify that the loaded terms matched the expected regional and tracer-space accounting.
- Set up the MATLAB utilities that later became the reference point for Python rewrites, including ASTE vector/tracer reshaping, grid-mask handling, and basic plotted checks.
- In research terms, 2023 is the starting point for proving that the model diagnostics could be loaded and balanced before moving into the larger Barents Sea WMT analysis.

### 2024 -- reproducing Lind and Pemberton style figures 

- Moved the initial MATLAB budget-loading and layer-closure ideas into a broader Python/Jupyter analysis environment.
- Built the foundation for ASTE/Barents Sea analysis: ASTE grid handling, cartopy mapping, basin and regional masks, bathymetry displays, gateway definitions, and reliable reading of monthly budget/state output.
- Developed the core T-S diagnostic machinery: bin definitions, volumetric distributions, state-variable binning, and early checks against model state output.
- Extended T-S binning into heat and salt tendency calculations, connecting volume distributions to tracer budgets rather than treating them as standalone plots.
- Worked through both physical-space and T-S/layer-space budget mechanics: advective fluxes, surface forcing, G terms, J terms, residual terms, and consistency between Eulerian budget terms and tracer-space transformations.
- Used single-cell, single-bin, and small-patch validation notebooks to isolate sign conventions, masking choices, surface forcing placement, and vector-flux behavior before trusting regional results.
- Began reproducing and adapting Pemberton-style and Lind-style Barents Sea figures, including T-S diagrams, boundary exchanges, cross-sections, salt budgets, WMT figures, and polar-front context.
- Added ECCO/PODAAC examples and Lab Sea tutorial-style diagnostics as external references for reading data, closing volume budgets, and testing T-S workflows outside the main Barents Sea setting.
- Started Overleaf proposal/manuscript material, tying the computational archive to the developing PhD proposal and Barents Sea science narrative.

### 2025

- Turned the 2024 mechanics into more complete Barents Sea story notebooks rather than isolated validation exercises.
- Built the main narrative diagnostics: trend maps, time series, Hovmollers, polar-front masks, mixed-layer depth analysis, gateway transports, Atlantic Water transport, Atlantic Water layer thickness, ocean heat content, sea-ice context, and interannual comparisons.
- Focused on 2007/2014/2016 contrasts to connect online and offline diagnostics with visible Barents Sea Atlantification signals.
- Expanded the gateway and transport work into a physical interpretation of how boundary exchanges, Atlantic inflow, regional storage, and surface forcing shape the Barents Sea heat/salt budget.
- Dug deeply into WMT and the G/J/M-term framework, including ASTE90/miniASTE and ASTE270 validation notebooks that test whether the tracer-space accounting closes at multiple scales.
- Created many focused M-term debugging notebooks, moving from one-cell and few-cell examples to small gateway sections and larger Barents Sea masks.
- Developed ASTE90 as a lower-cost validation environment before applying the same logic to ASTE270.
- Built presentation and proposal material, including fall 2025 presentation notebooks, year-comparison figures, and continued Overleaf edits.
- Expanded manuscript-style figure production for Barents Sea budgets, cross-sections, transports, polar-front structure, and Atlantification, using the earlier Pemberton/Lind replications as scientific and visual reference points.
- By the end of 2025, the archive had shifted from "can the terms be computed?" to "what Barents Sea story do the closed terms support?"

### 2026

- Developed a formal validation framework for tracer-space volume budgets, with explicit equation-by-equation checks rather than qualitative agreement between plots.
- Closed Eqs. 2-5 of the `pkg_layers` formulation in miniASTE and ASTE270, connecting layer-volume tendencies, advective terms, surface/boundary contributions, and transformation terms into one consistent accounting system.
- Organized the validation around readable entry points such as `FULL_MINIASTE_WALKTHROUGH.ipynb`, `validate_eq1_2_3.ipynb`, `validate_eq4.ipynb`, `validate_eq5.ipynb`, and `validate_alleqs_aste270.ipynb`.
- Identified and explained projection artifacts arising from face-centered versus cell-centered tracer binning, clarifying why some apparent mismatches came from how quantities were projected into tracer space rather than from physical budget failure.
- Developed a generalized interpretation linking gateway transports, ADVh, J-vectors, and Walin-style transformations, making the tracer-space budget easier to interpret physically instead of treating it only as a numerical closure exercise.
- Built reusable demonstrations for single-column, single-cell, small-region, full regional, ASTE90/miniASTE, ASTE270, and Lab Sea configurations.
- Extended the framework from volume closure into physically interpretable heat and salt budget decompositions, including surface heat flux, freshwater/salt forcing, sea-ice growth/melt effects, and salt-plume contributions.
- Used Fortran source snippets and Lab Sea experiments to trace model surface-forcing terms back to their implementation, especially for sea-ice and salt-plume diagnostics.
- Added cleaned surface-term breakdown notebooks for miniASTE, ASTE270, and Lab Sea cases so the same physical decomposition can be shown across model configurations.
- Consolidated the Barents Sea WMT manuscript workflow around validated layer budgets, gateway terms, surface transformations, and heat/salt interpretation.
- Established the framework now used for ongoing Barents Sea WMT manuscript development: the archive can demonstrate not only where the diagnostics are, but why the budget equations close and how the resulting terms map onto the physical Atlantification story.

## Good Entry Points

- For a broad Barents Sea narrative: start with `ASTE_270/Pemberton_BarentsSpaper/Fullstory_BS_postOct1/README.txt`, then `trends_maps.ipynb`, `timeseries_hovmoller.ipynb`, and `wmt_spellout_bigaste.ipynb`.
- For current validation work: start with `ASTE_270/Pemberton_BarentsSpaper/2026_BarentsS/validate_all_pkg_layers_miniaste/FULL_MINIASTE_WALKTHROUGH.ipynb`.
- For regional budgets across model configurations: start with `budgeting/regional_budgeting/FULLASTE_MULTI_RUN_SETUP.md`.
- For WAOM floats: start with `WAOM/Modifying_Floats/ID_WaterMasses.ipynb` and `WAOM/figure_scripts/create_all_plots.ipynb`.
- For reusable code: start with `an_helper_functions/` and `budgeting/regional_budgeting/fullaste_run_config.py`.
