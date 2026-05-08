# fullASTE Multi-Run Setup

The `fullASTE_budgeting_*.ipynb` notebooks repeat the same path, geometry, and
`myparms` setup. Use `fullaste_run_config.py` so changing runs only requires
editing one small cell.

## Replace The Path/Grid/Parameter Cells

In each `fullASTE...` notebook, replace the old cells that define:

- `dirroot`, `dirrun`, `runstr`, `dirIn`, `dirdiags`, `dirstate`, `dirGrid`
- `bigaste`, `labsea`, `oneface`, `nx`, `ny`, `nz`, `nfx`, `nfy`
- `save_budg_*`, `strbudg`, `kBudget`, `test3d`, `plot_fig`, `myparms`

with one setup cell like this:

```python
from fullaste_run_config import configure_import_paths, install_fullaste_config

configure_import_paths()

cfg = install_fullaste_config(
    globals(),
    profile="aste90",  # "aste270", "aste90", or "labsea"
    budget="Heat",     # "Mass", "Heat", or "Salt"
    dirroot="/scratch3/atnguyen/aste_90x150x60/",
    runstr="run_c68v_heffmosm3x_layers_lessmem1_viscAHp5em2_it0000_pk0000000001_r8_07May2026_pfe/",
)
```

For the Lab Sea runs:

```python
from fullaste_run_config import configure_import_paths, install_fullaste_config

configure_import_paths()

cfg = install_fullaste_config(
    globals(),
    profile="labsea",
    budget="Mass",
    dirroot="/scratch/mmurakami/labsea/layers/",
    dirGrid="/scratch2/atnguyen/labsea/GRID_r8/",
    runstr="run_c68v_layers_lessmem1_5ts_06May2026/",
    tsstr=["0000000003", "0000000004"],
)
```

For the original ASTE270 run:

```python
from fullaste_run_config import configure_import_paths, install_fullaste_config

configure_import_paths()

cfg = install_fullaste_config(
    globals(),
    profile="aste270",
    budget="Salt",
    dirroot="/scratch2/atnguyen/aste_270x450x180/",
    dirrun="/scratch/atnguyen/aste_270x450x180/OFFICIAL_ASTE_R1_Sep2019/",
)
```

## Optional Grid Loading Cleanup

The module also has helpers for repeated grid setup:

```python
from fullaste_run_config import load_mygrid, prepare_grid_metrics, apply_aste_open_boundary_mask

mygrid = load_mygrid(
    rdmds,
    dirGrid,
    nx=nx,
    ny=ny,
    nz=nz,
    oneface=oneface,
    facesSize=globals().get("facesSize"),
    facesExpand=globals().get("facesExpand"),
)

metrics = prepare_grid_metrics(mygrid, mk3D_mod, nz=nz, ny=ny, nx=nx)
globals().update(metrics)

if not oneface:
    hf = apply_aste_open_boundary_mask(
        mygrid["hFacC"],
        nx=nx,
        nz=nz,
        nfx=nfx,
        nfy=nfy,
        get_aste_tracer_func=get_aste_tracer,
        aste_tracer2compact_func=aste_tracer2compact,
    )
else:
    hf = mygrid["hFacC"].copy()
    hf[hf == 0] = np.nan
```

## Diagnostic File Lookup

Instead of hard-coding file prefixes like `budg3d_hflux_set2`, build a lookup
from the run's `namelists/data.diagnostics` or `NAMELISTS/data.diagnostics`:

```python
from fullaste_run_config import diagnostic_file_map, diagnostic_file_name

diag_to_file = diagnostic_file_map(layers_path)

file_name = diag_to_file["ADVx_TH"]       # "budg3d_hflux_set2"
file_name = diag_to_file["TFLUX"]         # "budg2d_zflux_set1"
file_name = diag_to_file["SALTDR"]        # "budg3d_snap_set2"

# Or query one diagnostic directly:
file_name = diagnostic_file_name(layers_path, "KPPg_TH")
```

Then use the result with the existing `dirIn`/`dirdiags` pattern:

```python
file_name = diagnostic_file_name(layers_path, "ADVx_TH")
ADVx_TH, its, meta = rdmds(
    os.path.join(dirdiags, file_name),
    t2,
    returnmeta=True,
    rec=recs[0],
)
```

## Why This Helps

- `strbudg` now follows the notebook budget instead of being copied from a
  different notebook.
- ASTE270, ASTE90, and Lab Sea geometry live in one profile table.
- Path names keep the old notebook variable names, so existing budget code can
  keep using `dirIn`, `dirdiags`, `dirstate`, `dirGrid`, `nx`, `ny`, `nz`, etc.
- Diagnostic file names can come from `data.diagnostics`, so runs with different
  stream names do not require editing budget logic.
- Run-specific edits are limited to `profile`, `budget`, `dirroot`, `runstr` or
  `dirrun`, and optional parameter overrides.
