"""Which vertical cut planes does the exact chart compiler refuse?

A vertical plane a x + b y <= d contains a vertical box edge exactly when d equals a*x0 + b*y0 for a corner
(x0, y0) of the unit square, that is d in {0, a, b, a+b}.  The trace of the cut on a port face then coincides with
an edge of that face.  This probe walks d across those values and records which ones compile.
"""
import sys
from fractions import Fraction
import numpy as np
from pathlib import Path
R = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1"); sys.path.insert(0, str(R / "src"))
from importlib import import_module
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
fcb = import_module(f"{P}.full_cube_backend"); fcg = import_module(f"{P}.full_cube_geometry")
from generated_cell.pred777h_global_p1_thickness_manifest import GlobalP1ThicknessField
from generated_cell.pred768_canonical_geometry_spec import CanonicalRationalPlane

TAU = Fraction(2, 5)
field = GlobalP1ThicknessField(field_id="PROBE", origin=(Fraction(0),) * 3, cell_size=(Fraction(1),) * 3, cell_shape=(1, 1, 1), vertex_values=(TAU,) * 8)
cases = []
for a, b, label in ((Fraction(1), Fraction(1), "45 deg"), (Fraction(1), Fraction(0), "0 deg"), (Fraction(2), Fraction(1), "26.6 deg")):
    for d in (Fraction(0), a / 2, a, (a + b) / 2, b if b else Fraction(1, 3), a + b - Fraction(1, 100), a + b, a + b + Fraction(1, 10)):
        cases.append((a, b, d, label))
for a, b, d, label in cases:
    try:
        spec = fcg.make_cell_spec(thickness_field=field, cell_world_origin=(Fraction(0),) * 3, cell_id="probe", parent_world_cut_id="CUT",
                                  world_cut=CanonicalRationalPlane("cut_0", "cut_0", a, b, Fraction(0), d))
        fcb.compile_full_cube_geometry_inputs(spec)
        status = "ok"
    except Exception as exc:
        status = f"{type(exc).__name__}: {str(exc)[:60]}"
    corner_values = sorted({Fraction(0), a, b, a + b})
    on_edge = d in corner_values
    print(f"{label:9s} a={float(a):.2f} b={float(b):.2f} d={float(d):.4f}  plane_contains_a_vertical_edge={on_edge}  -> {status}")
