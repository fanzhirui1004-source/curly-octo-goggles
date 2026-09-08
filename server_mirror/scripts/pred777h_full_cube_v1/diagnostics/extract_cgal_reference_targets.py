#!/usr/bin/env python3
"""Dump the Gmsh-route reference targets a CGAL cross-check compares against.

For each chosen cell: the exact geometry arguments the CGAL program needs, and the six uniform-strain apparent
stiffnesses of our produced label, which is the quantity both routes can compute without sharing a port space.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from importlib import import_module

import numpy as np

ROOT = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/pred777h_full_cube_v1/diagnostics"))
snap = import_module("pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.mesher_neutral_snapshot")
from label_error import load_label, uniform_strain_stiffness

POP = Path("/root/autodl-tmp/_claude_diag/population/cells")
LAB = Path("/root/autodl-tmp/_claude_diag/production/labels")
CELLS = ["pop_uncut_0529", "pop_uncut_0297", "pop_uncut_0658", "pop_cut_0136", "pop_cut_1215", "pop_cut_0473"]

out = []
for c in sys.argv[1:] or CELLS:
    _, spec, geom = snap.load_geometry_manifest(POP / c / "geometry_material_manifest.json")
    row = {"case": c,
           "tau_corners": [float(t) for t in geom.tau_corners],
           "cut_plane": None if geom.cut_plane is None else [float(v) for v in geom.cut_plane],
           "cgal_args": {"--tau-corners": ",".join("%.17g" % float(t) for t in geom.tau_corners),
                         "--collar": "1e-9",
                         **({"--cut": ",".join("%.17g" % float(v) for v in geom.cut_plane)} if geom.cut_plane is not None else {})}}
    d = LAB / c
    if (d / "FIXED_PORT_SCHUR_TET10.npz").exists():
        r = json.loads((d / "LABEL_RECEIPT.json").read_text())
        if r.get("status") == "PASS":
            lab = load_label(d); C = uniform_strain_stiffness(lab["S"], lab["xyz"])
            row["gmsh_route"] = {
                "resolution": r["resolution"], "q_active": r["schur"]["q_active"],
                "tet_count": r["mesh"]["tet_count"], "material_volume": r["mesh"]["material_volume"],
                "uniform_strain_stiffness": {k: float(v) for k, v in zip(("Exx", "Eyy", "Ezz", "Gyz", "Gxz", "Gxy"), C)},
                "max_eigenvalue": r["schur"]["max_eigenvalue"], "git_head": r["git_head"]}
    out.append(row)
print(json.dumps({"contract": {"E": 1.0, "nu": 0.3, "element": "TET10",
                               "quantity": "C[i] = u_i' S u_i with u_i the affine displacement of the i-th unit strain "
                                           "(xx, yy, zz, yz, xz, xy) imposed on the OUTER PORT FACES only; the TPMS free "
                                           "surface carries no condition; S is the Schur complement onto those faces",
                               "strain_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
                               "origin": "the centroid of the port nodes (the affine field is measured from it; a "
                                         "constant offset is a rigid translation and does not change the energy)"},
                  "cells": out}, indent=1))
