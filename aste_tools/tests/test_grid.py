from __future__ import annotations

import numpy as np

from aste_tools.grid import ASTEGridSpec, compact_to_tracer, compact_uv_to_tracer, tracer_to_compact


def test_compact_to_tracer_roundtrip_for_tiny_grid():
    grid = ASTEGridSpec(nx=4, ncut1=5, ncut2=3, nz=2)
    compact = np.arange(np.prod(grid.compact_shape_3d), dtype=float).reshape(grid.compact_shape_3d)

    tracer = compact_to_tracer(compact, grid)
    restored = tracer_to_compact(tracer, grid)

    assert tracer.shape == grid.tracer_shape_3d
    np.testing.assert_allclose(restored, compact)


def test_uv_to_tracer_adds_staggered_padding():
    grid = ASTEGridSpec(nx=4, ncut1=5, ncut2=3, nz=2)
    u = np.ones(grid.compact_shape_3d)
    v = np.ones(grid.compact_shape_3d) * 2

    up, vp = compact_uv_to_tracer(u, v, grid)

    assert up.shape == (2, grid.tracer_shape_2d[0] + 1, grid.tracer_shape_2d[1] + 1)
    assert vp.shape == up.shape
    assert np.isfinite(up).any()
    assert np.isfinite(vp).any()

