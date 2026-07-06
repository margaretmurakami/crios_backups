"""Run a no-data ASTE tooling demonstration.

This demo intentionally uses synthetic arrays so it can run on a laptop or login
node without access to the full ASTE scratch directories.
"""

from __future__ import annotations

import numpy as np

from aste_tools.binning import bin_array, create_ts_mesh
from aste_tools.grid import ASTEGridSpec, compact_to_tracer, compact_uv_to_tracer, tracer_to_compact
from aste_tools.masks import masked_area_mean
from aste_tools.timing import monthly_file_steps


def main() -> None:
    grid = ASTEGridSpec(nx=4, ncut1=5, ncut2=3, nz=2, name="tiny-demo")
    compact = np.arange(np.prod(grid.compact_shape_3d), dtype=float).reshape(grid.compact_shape_3d)

    tracer = compact_to_tracer(compact, grid)
    roundtrip = tracer_to_compact(tracer, grid)
    print(f"compact shape: {compact.shape}")
    print(f"tracer shape: {tracer.shape}")
    print(f"roundtrip max error: {np.nanmax(np.abs(compact - roundtrip)):.1f}")

    u_plot, v_plot = compact_uv_to_tracer(compact, compact + 1000, grid)
    print(f"u/v plotting shapes: {u_plot.shape}, {v_plot.shape}")

    salt_edges = np.array([33.0, 34.0, 35.0, 36.0])
    temp_edges = np.array([-2.0, 0.0, 2.0, 4.0])
    salinity = np.array([[[33.5, 34.5], [35.5, np.nan]]])
    theta = np.array([[[-1.0, 1.0], [3.0, 10.0]]])
    volume = np.array([[[2.0, 3.0], [4.0, 5.0]]])
    meshes = create_ts_mesh(
        volume,
        bin_array(salinity, salt_edges),
        bin_array(theta, temp_edges),
        n_salinity=3,
        n_theta=3,
    )
    print(f"T-S mesh total: {np.nansum(meshes):.1f}")

    mean, area = masked_area_mean(compact[0], np.ones(grid.compact_shape_2d), np.ones(grid.compact_shape_2d))
    print(f"masked mean: {mean:.1f}; area: {area:.0f}")

    steps = monthly_file_steps(delta_t=3600, start_year=1979, end_year=1980)
    print(f"first three monthly file steps: {steps[:3].tolist()}")


if __name__ == "__main__":
    main()

