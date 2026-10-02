#!/usr/bin/env python3
"""Figures 3 (F04_correction), 8 (F03_spectrum) and Supplementary Figures S03-S05
(S02A_smoothing, S02B_coarse_spaces, S03_sensitivity_diagnostics).

Copied from the round-2 audit build (04_figures/audit2/scripts/build_publication.py, functions
f03, f04, s03a, s03b, s04). Plotting code and data selection are unchanged; only visible labels
follow the R3 terminology (energy error, cumulative energy fraction, coarse DOFs, interior, variant
names). Inputs: the four JSON tables in ./tables (verbatim copies of 03_results/tables, SHA-256 checked
below). No solver, training, geometry generation or network access is used.

Run from anywhere:  python3 docs/paper_p1/figures_src/codex/build_publication.py [--only f03 f04 s03a s03b s04]
Writes docs/paper_p1/figures/<name>.{png,pdf,svg} (PNG at 300 dpi) and provenance under ./_build.
"""
from __future__ import annotations
import hashlib
import io
import argparse
import importlib.util
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
OUT = Path(os.environ.get("P1_FIG_OUT", HERE.parents[1] / "figures"))
DATA = HERE / "_build" / "data"
QA = HERE / "_build" / "qa"
# SHA-256 of the input tables as used for the round-2 figures (audit2 SOURCE_MANIFEST / bindings).
EXPECTED_TABLES = {
    "03_smoothing": "7c81c3a93462051f371f18a63d47b3e56e709c1d5e9571e6d0076249d61f0060",
    "04_coarse_correction": "f9c006eb85cf00bcbcfe5517eb9e775bc30b8a29b45256e6e1f5ef911a3fd257",
    "05_spectrum": "cfabafdbe292daf99c5ef06fff9493fd4753f4cca2cb5d4407f16388ac7e5db1",
    "06_sensitivity_diagnostic": "702b05aac7386b1de96bc5469a576135d244f879356d6647ca39bf57e862e922",
}
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "_build" / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PolyCollection, LineCollection
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
from PIL import Image, ImageOps

for folder in (OUT, DATA, QA):
    folder.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
    "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
    "svg.hashsalt": "p1-round2", "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": .6,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.unicode_minus": True})

C = dict(ink="#243447", muted="#657382", grid="#DFE5E9", gray="#343D46",
         blue="#0072B2", orange="#D55E00", green="#009E73", purple="#AA4499", gold="#E69F00")
# Palette (scheme A), mirroring figstyle.py: layer 1 variants, layer 2 cells (blue ramp by cut severity, shape by
# stratum), layer 3 other categories (greys; the only accent is NICE vermillion for the complete NICE correction).
VAR = dict(base="#8A94A0", uncorrected="#0072B2", nice="#D55E00")
GREY = ("#243447", "#657382", "#A3ADB8"); GREY_FILL = "#CDD3DA"
CELL = {"U1": ("#8FB1D6", "o"), "U2": ("#7AA2CD", "o"), "L1": ("#6390C2", "D"), "M1": ("#4A7AB0", "s"),
        "M2": ("#3A679C", "D"), "H1": ("#24528A", "^"), "H2": ("#133A68", "v"), "H3": ("#0B2747", "<")}
CASES = ["fresh_val_2000_full", "fresh_val_2003_d1_v1", "fresh_val_2005_d1_v0", "fresh_val_2006_d0_v1", "fresh_val_2010_d0_v0"]
_CELL_OF = dict(zip(CASES, ["U1", "M1", "H1", "M2", "H2"]))
CASE_COLOR = {c: CELL[k][0] for c, k in _CELL_OF.items()}
CASE_MARK = {c: CELL[k][1] for c, k in _CELL_OF.items()}
SEVERITY = [CASES[0], CASES[1], CASES[3], CASES[2], CASES[4]]  # U1, M1, M2, H1, H2: legend order follows the ramp
CLASS_LABEL=dict(force="Nodal force",support="Spring support",face="Single-face force",macro="Polynomial",grf="Multiscale",force_c="Traction",face_c="Face traction",support_k="Stiffness support",glued="Neighbour-induced")
TABLE_CACHE, TABLE_HASHES, SOURCE_HASHES, BUILD_LOG = {}, {}, {}, {}
CURRENT, ROWS, BINDINGS = None, {}, []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def table(name):
    if name not in TABLE_CACHE:
        f = TABLES / f"{name}.json"
        TABLE_HASHES[name] = sha(f)
        assert TABLE_HASHES[name] == EXPECTED_TABLES[name], "Input table differs from the round-2 table: "+name
        doc = json.loads(f.read_text())
        assert doc["schema_version"] == "p1-results-1.0"
        records = doc["rows"]
        assert len({r["record_id"] for r in records}) == len(records), name
        for row in records:
            row["_table"] = name
        TABLE_CACHE[name] = records
    return TABLE_CACHE[name]


def select(name, **filters):
    return [r for r in table(name) if all(r.get(k) == v for k, v in filters.items())]


def one(name, **filters):
    r = select(name, **filters)
    assert len(r) == 1, (name, filters, len(r))
    return r[0]


def begin(name):
    global CURRENT, ROWS, BINDINGS
    CURRENT, ROWS, BINDINGS = name, {}, []


def bind(records, panel, role, scale=100, finite=True):
    ids = []
    for r in records:
        if finite:
            assert r["value_status"] == "finite" and np.isfinite(r["value"]), r["record_id"]
        key = r["_table"]+":"+r["record_id"]
        ROWS[key] = r
        ids.append(key)
        source = r["source"]  # raw evidence files are not shipped here; the table hash above pins the data
        SOURCE_HASHES.setdefault(source["file"], source["sha256"])
        assert SOURCE_HASHES[source["file"]] == source["sha256"]
    BINDINGS.append(dict(panel=panel, role=role, row_ids=ids, display_scale=scale))
    return np.array([r["value"]*scale if r["value"] is not None else np.nan for r in records])


def short(case):
    return dict(fresh_val_2000_full="U1",fresh_val_2001_full="U2",fresh_val_2003_d1_v1="M1",fresh_val_2006_d0_v1="M2",fresh_val_2005_d1_v0="H1",fresh_val_2010_d0_v0="H2").get(case,case.replace("fresh_val_", "").replace("fresh_train_", "").replace("_", "/"))


def canvas(title, subtitle, rows, cols, height=156, top=.80, bottom=.11, hspace=.57, wspace=.36):
    specs={"F02_validation":(122,.87,.11),"F03_spectrum":(122,.87,.11),
      "F04_correction":(131,.84,.12),"F05_assembly":(157,.90,.115),
      "F06_bernstein":(79,.82,.17),"F07_cost":(131,.88,.205),
      "S02_distributions":(173,.915,.08),"S02A_smoothing":(129,.85,.11),
      "S02B_coarse_spaces":(193,.89,.09),"S03_sensitivity_diagnostics":(129,.875,.175),
      "S05_iterative_solves":(177,.88,.085)}
    height,top,bottom=specs.get(CURRENT,(height,top,bottom))
    fig=plt.figure(figsize=(180/25.4,height/25.4))
    fig._scientific_title=title
    fig._setting_note=subtitle
    grid=fig.add_gridspec(rows,cols,left=.10,right=.975,bottom=bottom,top=top,
                         hspace=hspace,wspace=wspace)
    return fig,grid


def style(ax, title=None, xlabel=None, ylabel=None, logy=False):
    if title:
        ax.set_title(("("+title[0]+") "+title[3:]) if len(title)>2 and title[1:3]=="  " else title, loc="left", weight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log"); ax.yaxis.set_minor_locator(NullLocator())
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["muted"])
    ax.tick_params(colors=C["ink"], length=3, width=.6)
    ax.grid(axis="y", color=C["grid"], lw=.35)
    ax.set_axisbelow(True)


def case_handles():
    return [Line2D([], [], c=CASE_COLOR[c], marker=CASE_MARK[c], ms=4, lw=1.1, label=short(c)) for c in SEVERITY]


def legend(fig, handles, ncol=3, y=.885, size=7.5):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.085, 1.005), ncol=ncol,
               frameon=False, fontsize=size, columnspacing=1.3, handlelength=1.7, handletextpad=.5)


def step_axis(ax):
    ax.set_xscale("symlog", base=2, linthresh=1, linscale=1)
    ax.xaxis.set_major_locator(FixedLocator([0,1,2,4,8,16,32]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlim(-.1, 38)


def odd_decades(ax):
    """Label every second decade (10^-3, 10^-1, ...), as the round-2 build rendered these wide log axes."""
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([10.0**e for e in range(-15, 16, 2) if lo <= 10.0**e <= hi]))


def visible_texts(fig):
    texts = list(fig.texts)
    for leg in fig.legends:
        texts.extend(leg.get_texts())
    for ax in fig.axes:
        texts.extend(ax.texts)
        texts.extend([ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label])
        leg = ax.get_legend()
        if leg:
            texts.extend(leg.get_texts())
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks():
                if lo-1e-12 <= tick.get_loc() <= hi+1e-12:
                    texts.extend([tick.label1, tick.label2])
    return [t for t in texts if t.get_visible() and t.get_text()]


def finish(fig, note="", extra=None):
    fig.canvas.draw()
    renderer, frame = fig.canvas.get_renderer(), fig.bbox
    outside = []
    for t in visible_texts(fig):
        b = t.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1 > frame.x1+1 or b.y1 > frame.y1+1):
            outside.append(t.get_text())
    numeric_outside, checked_vertices = [], 0
    def check_vertices(ax, values, label, check_x=True, check_y=True):
        nonlocal checked_vertices
        values = np.asarray(values, dtype=float).reshape(-1, 2)
        values = values[np.isfinite(values).all(axis=1)]
        checked_vertices += len(values)
        if not len(values):
            return
        for column, limits, enabled in [(0, ax.get_xlim(), check_x), (1, ax.get_ylim(), check_y)]:
            if not enabled:
                continue
            lo, hi = sorted(limits)
            tol = max(abs(lo), abs(hi), 1) * 2e-6
            bad = (values[:, column] < lo-tol) | (values[:, column] > hi+tol)
            if bad.any():
                numeric_outside.append(dict(axis=fig.axes.index(ax), artist=label,
                    coordinate="xy"[column], limits=[lo, hi], values=values[bad, column].tolist()))
    for ax in fig.axes:
        if ax.name == "3d":
            continue  # 3-D geometry vertices are checked in geometry_figure before projection.
        for k, line in enumerate(ax.lines):
            transform = line.get_transform()
            if transform == ax.transData:
                check_vertices(ax, line.get_xydata(), f"line_{k}")
            elif transform == ax.get_xaxis_transform():
                check_vertices(ax, line.get_xydata(), f"vertical_reference_{k}", check_y=False)
            elif transform == ax.get_yaxis_transform():
                check_vertices(ax, line.get_xydata(), f"horizontal_reference_{k}", check_x=False)
        for k, col in enumerate(ax.collections):
            if isinstance(col, PathCollection) and col.get_offset_transform() == ax.transData:
                check_vertices(ax, col.get_offsets(), f"scatter_{k}")
            elif isinstance(col, PolyCollection) and col.get_transform() == ax.transData:
                for path in col.get_paths():
                    check_vertices(ax, path.vertices, f"band_{k}")
            elif isinstance(col, LineCollection) and col.get_transform() == ax.transData:
                for segment in col.get_segments():
                    check_vertices(ax, segment, f"interval_{k}")
        for k, container in enumerate(ax.containers):
            if isinstance(container, matplotlib.container.BarContainer):
                for rect in container.patches:
                    b = rect.get_bbox()
                    check_vertices(ax, [[b.x0,b.y0],[b.x1,b.y1]], f"bar_{k}")
    assert not numeric_outside, (CURRENT, numeric_outside)
    assert not outside, (CURRENT, outside)
    for ext in ["svg", "pdf", "png"]:
        buffer = io.BytesIO()
        fig.savefig(buffer, format=ext, dpi=300)
        (OUT / f"{CURRENT}.{ext}").write_bytes(buffer.getvalue())
    with Image.open(OUT / f"{CURRENT}.png") as im:
        ImageOps.grayscale(im).save(QA / f"{CURRENT}_gray.png")
    tags = [e.tag.split("}")[-1] for e in ET.parse(OUT / f"{CURRENT}.svg").iter()]
    assert tags.count("image") == 0
    sources = sorted({r["source"]["file"] for r in ROWS.values()})
    write_json(DATA / f"{CURRENT}_bindings.json", dict(
        figure=CURRENT, table_inputs={r["_table"]:TABLE_HASHES[r["_table"]] for r in ROWS.values()},
        source_files={s:SOURCE_HASHES[s] for s in sources},
        rows=list(ROWS.values()), bindings=BINDINGS,
        extra=dict(extra or {}, scientific_title=getattr(fig,"_scientific_title",""), setting_note=getattr(fig,"_setting_note",""), caption_note=note), note="All plotted numerical values originate in task-3 tables. Display scale 100 converts ratios to percent."))
    BUILD_LOG[CURRENT] = dict(width_mm=round(fig.get_figwidth()*25.4,3),
        height_mm=round(fig.get_figheight()*25.4,3), svg_text_nodes=tags.count("text"),
        svg_raster_images=tags.count("image"), bound_rows=len(ROWS),
        bound_series=len(BINDINGS), text_outside_canvas=outside,
        checked_numeric_vertices=checked_vertices, numeric_outside_axes=numeric_outside)
    plt.close(fig)
    print(CURRENT, "rows", len(ROWS), "outside", outside)


def f03():
    begin("F03_spectrum")
    fig,grid=canvas("Spectral location of extension error", r"$Av=\lambda Dv,\quad D=\mathrm{diag}(A)$  ·  cumulative energy in the lowest modes",2,2,height=154)
    ERR,FLD,KEY=GREY[0],GREY[2],GREY[1]  # base-network error: ink filled; exact field: light grey open; style keys: mid grey
    handles=[Line2D([],[],color=ERR,marker="o",ms=3,lw=1.2,label="Error (filled)"),Line2D([],[],color=FLD,marker="o",mfc="white",ms=3,lw=1.2,label="Exact field (open)"),
             Line2D([],[],color=KEY,marker="o",lw=1,ms=3,label="Traction"),Line2D([],[],color=KEY,marker="^",ls="--",lw=1,ms=3,label="Nodal force")]
    legend(fig,handles,4)
    cases=[CASES[0],CASES[1],CASES[3],CASES[4]]
    for j,case in enumerate(cases):
        ax=fig.add_subplot(grid[j//2,j%2]);panel="abcd"[j]
        for field,col in [("error",ERR),("field",FLD)]:
            for cls,ls,mark in [("force_c","-","o"),("force","--","^")]:
                rr=sorted(select("05_spectrum",case=case,field=field,direction_class=cls,statistic="mean"),key=lambda r:r["modes"])
                x=[r["modes"] for r in rr]
                ax.plot(x,bind(rr,panel,f"{field}/{cls}/mean"),c=col,ls=ls,marker=mark,mfc=col if field=="error" else "white",ms=3,lw=1)
                if cls=="force_c":
                    lower=[one("05_spectrum",case=case,field=field,direction_class=cls,statistic="p10",modes=m) for m in x]
                    upper=[one("05_spectrum",case=case,field=field,direction_class=cls,statistic="p90",modes=m) for m in x]
                    ax.fill_between(x,bind(lower,panel,f"{field}/p10"),bind(upper,panel,f"{field}/p90"),color=col,alpha=.13,lw=0)
        ax.set_xscale("log");ax.set_xticks([1,10,50,200],["1","10","50","200"]);ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1,220);ax.set_ylim(0,40)
        style(ax,panel+"  "+short(case),"Cumulative mode count","Cumulative energy fraction (%)")
    finish(fig,"Bands: 10th–90th directional percentiles for force_c. Error and exact field use their own interior-energy denominators.")


def f04():
    begin("F04_correction")
    fig,grid=canvas("Equilibrium correction of learned extensions", "v2L1  ·  force_c bank  ·  fixed retained trace",2,2,height=165,top=.785,wspace=.39,hspace=.68)
    legend(fig,case_handles(),3)
    for j,metric in enumerate(["energy_mean","sens_mean"]):
        ax=fig.add_subplot(grid[0,j]);panel="ab"[j]
        for case in CASES:
            rr=sorted(select("03_smoothing",case=case,direction_class="force_c",start="net",metric=metric),key=lambda r:r["k"])
            ax.plot([r["k"] for r in rr],bind(rr,panel,case),c=CASE_COLOR[case],marker=CASE_MARK[case],ms=3.4,lw=1.1)
        step_axis(ax);ax.set_ylim(.001,150)
        style(ax,panel+"  "+["Energy error","Sensitivity error"][j],"Chebyshev steps, k",["Mean energy error (%)","Mean sensitivity error (%)"][j],True)
    ax=fig.add_subplot(grid[1,0]);case=CASES[1]
    methods=["net","net_tail","net+Q1_17+tail","net+tail+Q1_17+tail"]
    labels=["Network\nonly\n0 total steps","Post-8\nsmoothing\n8 total steps","$Q_1(17)$\n→ Post-8\n8 total steps","Pre-8 → $Q_1(17)$\n→ Post-8\n16 total steps"]
    # correction stages: greys (layer 3); only the complete correction (pre-8 -> Q1(17) -> post-8) is vermillion
    for i,(method,color,mark) in enumerate(zip(methods,[GREY[0],GREY[1],GREY[2],VAR["nice"]],["o","D","^","s"])):
        rr=[one("04_coarse_correction",case=case,direction_class="force_c",k_per_tail=8,method=method,statistic=s) for s in ["mean","p90"]]
        vals=bind(rr,"c",method)
        ax.plot([i,i],vals,c=color,lw=1);ax.plot(i,vals[0],mark,c=color,ms=4.5);ax.plot(i,vals[1],"_",c=color,ms=8)
        ax.annotate(f"{vals[0]:.3g}%",(i,vals[0]),xytext=(0,-13),textcoords="offset points",ha="center",fontsize=7)
    ax.set_xticks(range(4),labels,fontsize=6.5);ax.set_xlim(-.4,3.4);ax.set_ylim(.055,40)
    style(ax,"c  M1: correction stages",None,"Energy error (%)",True)
    ax=fig.add_subplot(grid[1,1])
    for method,label,color,marker in [("net_tail","One smoothing stage",GREY[1],"D"),("net+tail+Q1_17+tail","Two-stage + $Q_1(17)$",VAR["nice"],"s")]:
        rr=[one("04_coarse_correction",case=case,direction_class="force_c",k_per_tail=k,method=method,statistic="mean") for k in [2,4,8]]
        ax.plot([2,4,8],bind(rr,"d",method),c=color,marker=marker,ms=4,lw=1.1,label=label)
    ax.set_xticks([2,4,8]);ax.set_xlim(1,9);ax.set_ylim(.1,20)
    style(ax,"d  M1: smoothing budget","Steps per smoothing stage, k","Mean energy error (%)",True)
    ax.legend(frameon=False,fontsize=7,loc="lower left")
    finish(fig,"Panel c uses four methods from the same archived coarse-correction run. Total smoothing steps: 0, 8, 8, 16. Equal step counts do not imply equal cost. Q1_17 has 5,601 coarse DOFs. Panel d unchanged: two-stage correction uses k steps before and after the coarse solve.")


def s03a():
    begin("S02A_smoothing")
    fig,grid=canvas("Smoothing from learned and zero interior fields", "Base network  ·  fixed retained values  ·  α = 30",2,2,height=168,top=.79,hspace=.6)
    legend(fig,case_handles(),3)
    for i,metric in enumerate(["energy_mean","sens_mean"]):
        for j,cls in enumerate(["force_c","force"]):
            ax=fig.add_subplot(grid[i,j]);panel="abcd"[i*2+j]
            for case in CASES:
                for start,ls in [("net","-"),("zero","--")]:
                    all_rows=sorted(select("03_smoothing",case=case,direction_class=cls,start=start,metric=metric),key=lambda r:r["k"])
                    missing=[r for r in all_rows if r["value_status"]!="finite"]
                    if missing:bind(missing,panel,f"{case}/{start}/unrecorded",finite=False)
                    rr=[r for r in all_rows if r["value_status"]=="finite"]
                    ax.plot([r["k"] for r in rr],bind(rr,panel,f"{case}/{start}"),c=CASE_COLOR[case],ls=ls if len(rr)>1 else "",marker=CASE_MARK[case],ms=3.2,lw=1,
                            markerfacecolor=CASE_COLOR[case] if start=="net" else "white")
            step_axis(ax);ax.set_ylim(.001,2e6 if i==0 else 3e4)
            style(ax,panel+"  "+CLASS_LABEL[cls]+": "+["energy","sensitivity"][i],"Chebyshev steps, k",["Mean energy error (%)","Mean sensitivity error (%)"][i],True)
            odd_decades(ax)
    finish(fig,"Solid / filled: learned initial field. Dashed / open: zero interior field. Zero-start sensitivity is recorded only at k = 32.")


def s03b():
    begin("S02B_coarse_spaces")
    fig,grid=canvas("Interior coarse-space comparisons", "Base network  ·  eight steps per smoothing stage  ·  dots: means; caps: directional p90",3,2,height=225,top=.81,bottom=.13,hspace=.68,wspace=.4)
    spaces=["Q1_9","PU_9","Q1_17","Q2_17","PU_17","Q1_33"]
    # sequences: greys (layer 3); the complete correction in vermillion; zero start in light grey
    recipes=[("net+{s}","Network + coarse",GREY[0],"o"),
             ("net+{s}+tail","Network + coarse + post",GREY[1],"^"),
             ("net+tail+{s}+tail","Network + pre + coarse + post",VAR["nice"],"s"),
             ("zero+tail+{s}+tail","Zero + pre + coarse + post",GREY[2],"D")]
    hs=[Line2D([],[],c=col,marker=mark,ls="",ms=4,label=lab) for _,lab,col,mark in recipes]
    hs.extend([Line2D([],[],c=GREY[1],ls="--",lw=1,label="Network initial field"),Line2D([],[],c=GREY[1],ls=":",lw=1,label="Network + one smoother")])
    legend(fig,hs,2,y=.90,size=7.3)
    cases=[CASES[0],CASES[1],CASES[3]]
    for i,case in enumerate(cases):
        for j,cls in enumerate(["force_c","force"]):
            panel="abcdef"[2*i+j];ax=fig.add_subplot(grid[i,j]);dofs=[]
            for x,s in enumerate(spaces):
                for n,(pattern,lab,col,mark) in enumerate(recipes):
                    method=pattern.format(s=s)
                    rr=[one("04_coarse_correction",case=case,direction_class=cls,k_per_tail=8,method=method,statistic=stat) for stat in ["mean","p90"]]
                    y=bind(rr,panel,method);xp=x+(n-1.5)*.15
                    ax.plot([xp,xp],y,c=col,lw=.6);ax.plot(xp,y[0],mark,c=col,ms=3.3);ax.plot(xp,y[1],"_",c=col,ms=4)
                    if n==0:dofs.append(rr[0]["coarse_dofs"])
            for method,ls in [("net","--"),("net_tail",":")]:
                rr=[one("04_coarse_correction",case=case,direction_class=cls,k_per_tail=8,method=method,statistic="mean")]
                ax.axhline(bind(rr,panel,method)[0],ls=ls,c=GREY[1],lw=.8)
            ax.set_xticks(range(6),[("$Q_"+s[1]+"("+s.split("_")[1]+")$" if s.startswith("Q") else "PU("+s.split("_")[1]+")")+"\n"+f"{d:,}" for s,d in zip(spaces,dofs)],fontsize=7)
            ax.set_xlim(-.55,5.55);ax.set_ylim(.001,2e6)
            style(ax,panel+"  "+short(case)+" · "+CLASS_LABEL[cls],"Coarse family / coarse DOFs","Energy error (%)",True)
            odd_decades(ax)
    finish(fig,"Pre and post stages each contain eight smoothing steps. Coarse DOFs act only inside the cell; the assembly trace is unchanged.")


def s04():
    begin("S03_sensitivity_diagnostics")
    fig,grid=canvas("Field-based sensitivity diagnostics", "Matched trace inputs  ·  Base network and Uncorrected  ·  eight-component design response",2,2,height=166,top=.82,bottom=.17,hspace=.59,wspace=.4)
    hs=[Line2D([],[],c=VAR["base"],marker="o",ls="",ms=4,label="Base network (filled)"),Line2D([],[],c=VAR["uncorrected"],marker="o",mfc="white",ls="",ms=4,label="Uncorrected (open / hatched)"),
        Line2D([],[],c=GREY[0],marker="o",mfc="none",ls="",ms=4,label="Traction"),Line2D([],[],c=GREY[0],marker="^",mfc="none",ls="",ms=4,label="Nodal force")]
    legend(fig,hs,4)
    ax=fig.add_subplot(grid[0,0])
    for model,col in [("v2L1",VAR["base"]),("A0_ctrl",VAR["uncorrected"])]:
        for cls,mark in [("force_c","o"),("force","^")]:
            rr=sorted(select("06_sensitivity_diagnostic",model=model,direction_class=cls,metric="energy_excess_mean"),key=lambda r:r["case"])
            ss=[one("06_sensitivity_diagnostic",model=model,direction_class=cls,metric="sens_rel_mean",case=r["case"]) for r in rr]
            x,y=bind(rr,"a",model+"/"+cls+"/energy"),bind(ss,"a",model+"/"+cls+"/sensitivity")
            ax.scatter(x,y,facecolors=col if model=="v2L1" else "white",edgecolors=col,marker=mark,s=22,lw=.8)
    ax.set_xscale("log");ax.xaxis.set_minor_locator(NullLocator());ax.set_xlim(.02,80);ax.set_ylim(.02,150)
    style(ax,"a  Energy and sensitivity","Mean energy error (%)","Mean sensitivity error (%)",True)
    cases=sorted({r["case"] for r in select("06_sensitivity_diagnostic",model="v2L1")})
    ax=fig.add_subplot(grid[0,1])
    for model,col,offset in [("v2L1",VAR["base"],-.15),("A0_ctrl",VAR["uncorrected"],.15)]:
        for i,case in enumerate(cases):
            rr=select("06_sensitivity_diagnostic",model=model,direction_class="force_c",metric="first_order_share",case=case)
            if rr:ax.bar(i+offset,bind(rr,"b",model+"/"+case)[0],width=.28,color=col if model=="v2L1" else "white",hatch="///" if model=="A0_ctrl" else "",edgecolor=col if model=="A0_ctrl" else "white",lw=.3 if model=="v2L1" else .6)
    ax.set_xticks(range(len(cases)),[short(c) for c in cases]);ax.set_ylim(0,100)
    style(ax,"b  Linear-term norm share","Cell","Linear-term share (%)")
    groups=["vf<0.1","vf0.1-0.5","vf0.5-0.999","vf_full"]
    colors=[GREY[0],GREY[1],GREY[2],GREY_FILL];hatches=["///","..","xx",""]  # volume-fraction groups: greys + hatches
    for j,metric in enumerate(["group_abs_share","element_fraction"]):
        ax=fig.add_subplot(grid[1,j]);panel="cd"[j];bottom=np.zeros(len(cases))
        for group,col,hatch in zip(groups,colors,hatches):
            rr=[one("06_sensitivity_diagnostic",model="v2L1",direction_class="force_c",metric=metric,case=case,group_or_corner=group) for case in cases]
            yy=bind(rr,panel,group)
            ax.bar(range(len(cases)),yy,bottom=bottom,width=.64,color=col,hatch=hatch,edgecolor="white",lw=.4,label=group)
            bottom+=yy
        assert np.allclose(bottom,100,atol=2e-4), (metric,bottom)
        ax.set_xticks(range(len(cases)),[short(c) for c in cases]);ax.set_ylim(0,100)
        style(ax,panel+"  "+["Absolute error contributions","Element population"][j],"Cell","Share (%)")
    handles=[plt.Rectangle((0,0),1,1,fc=col,hatch=h,ec="white",label=lab) for col,h,lab in zip(colors,hatches,["vf < 0.1","0.1 ≤ vf < 0.5","0.5 ≤ vf < 0.999","vf ≥ 0.999"])]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.05),ncol=4,frameon=False,fontsize=7)
    finish(fig,"Panels b–d: force_c. The four volume-fraction groups are mutually exclusive; first-order share uses a separate norm definition.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',nargs='+',help='Rebuild selected function IDs only.')
    args=parser.parse_args()
    makers=[f03,f04,s03a,s03b,s04]
    if args.only:
        assert set(args.only)<=set(f.__name__ for f in makers)
    for make in makers:
        if not args.only or make.__name__ in args.only:make()
    for name,value in TABLE_HASHES.items():
        assert sha(TABLES/(name+".json"))==value, "Input table changed during build: "+name
    write_json(QA/"BUILD_CHECKS.json",dict(figures=BUILD_LOG,matplotlib=matplotlib.__version__,numpy=np.__version__,
        tables_verified=len(TABLE_HASHES),new_mechanics_computation=False))
    write_json(DATA/"SOURCE_MANIFEST.json",dict(table_hashes=TABLE_HASHES,source_hashes_from_table_rows=SOURCE_HASHES,
        figure_ids=list(BUILD_LOG),origin="04_figures/audit2/scripts/build_publication.py"))
    print("Completed",len(BUILD_LOG),"figures",len(TABLE_HASHES),"tables")


if __name__=="__main__":
    main()
