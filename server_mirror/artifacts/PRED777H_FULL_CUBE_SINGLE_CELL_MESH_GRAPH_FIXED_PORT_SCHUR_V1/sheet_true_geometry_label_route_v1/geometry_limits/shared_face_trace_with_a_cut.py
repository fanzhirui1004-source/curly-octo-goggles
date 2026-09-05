"""Do two neighbouring cells sharing ONE world cut plane agree on their shared port face?

Cell A occupies world [0,1]^3, cell B world [1,2]x[0,1]^2, and one world plane a*X + b*Y <= d cuts both.  The spec
carries the plane in world coordinates (chart compilation); the mesher geometry carries it in the cell's local frame
(d_local = d - a * offset).  The shared face is A's box_x_max and B's box_x_min; the test compares, in WORLD
coordinates, the carrier node set of the clipped trace there and its triangulation.  If the two disagree, cut cells
cannot be assembled from independently produced labels and the route needs a shared-trace rule.
"""
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
R = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1"); sys.path.insert(0, str(R / "src"))
from importlib import import_module
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
fcb = import_module(f"{P}.full_cube_backend"); fcg = import_module(f"{P}.full_cube_geometry")
pipe = import_module(f"{P}.sheet_label_pipeline"); cpm = import_module(f"{P}.conforming_port_mesh")
adapter = import_module(f"{P}.cgal_mesh3_adapter")
from generated_cell.pred777h_global_p1_thickness_manifest import GlobalP1ThicknessField
from generated_cell.geometry_native_exact_port_chart import *  # noqa
CanonicalRationalPlane = import_module("generated_cell.implicit_solid_stratified_topology").CanonicalRationalPlane if False else None
from importlib import import_module as _im
for modname in ("generated_cell.implicit_solid_stratified_topology", "generated_cell.pred777h_global_p1_thickness_manifest", "generated_cell.geometry_native_exact_port_chart"):
    m = _im(modname)
    if hasattr(m, "CanonicalRationalPlane"):
        CanonicalRationalPlane = m.CanonicalRationalPlane; break
print("plane class:", CanonicalRationalPlane)

TAU = Fraction(2, 5)
field = GlobalP1ThicknessField(field_id="PROBE", origin=(Fraction(0),) * 3, cell_size=(Fraction(1),) * 3, cell_shape=(2, 1, 1),
                               vertex_values=tuple([TAU] * 12))
a, b = Fraction(4, 5), Fraction(3, 5)
for d_world in (Fraction(13, 10), Fraction(3, 2), Fraction(17, 10)):
    cut = CanonicalRationalPlane("cut_0", "cut_0", a, b, Fraction(0), d_world)
    res = {}
    for tag, off in (("A", Fraction(0)), ("B", Fraction(1))):
        spec = fcg.make_cell_spec(thickness_field=field, cell_world_origin=(off, Fraction(0), Fraction(0)), cell_id=f"cell{tag}",
                                  parent_world_cut_id="CUT_SHARED", world_cut=cut)
        tau_local = tuple(float(field.value(spec.placement.world(np.array(c, dtype=float)))) for c in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        geom = adapter.CgalGeometryParameters(tau_local, (float(a), float(b), 0.0, float(d_world - a * off)))
        charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
        layout = pipe.build_carrier_layout(charts, spec, carrier_n=32)
        face = "box_x_max" if tag == "A" else "box_x_min"
        ports = tuple(sorted(str(t.source_id) for t in layout.local_traces))
        coords, _, per_face = cpm.carrier_shell(layout, ports)
        tri = dict(per_face).get(face)
        shift = float(off)
        if tri is None or not len(tri):
            res[tag] = (set(), set(), 0); continue
        gw = coords.copy(); gw[:, 0] += shift
        key = lambda p: tuple(round(float(v), 9) for v in p)
        nodes = np.unique(tri)
        res[tag] = ({key(gw[i]) for i in nodes}, {tuple(sorted(key(gw[i]) for i in t)) for t in tri}, len(tri))
    (nA, tA, cA), (nB, tB, cB) = res["A"], res["B"]
    print(f"world d={float(d_world):.2f}: cut meets the shared face at y={float((d_world - a) / b):.4f}")
    print(f"   A box_x_max: {cA} triangles, {len(nA)} nodes | B box_x_min: {cB} triangles, {len(nB)} nodes")
    print(f"   nodes only in A {len(nA - nB)} | only in B {len(nB - nA)} | triangles only in A {len(tA - tB)} | only in B {len(tB - tA)}")
    for name, s in (("A-only node", nA - nB), ("B-only node", nB - nA)):
        if s:
            print("   example", name, sorted(s)[:2])
