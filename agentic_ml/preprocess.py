"""Automated scRNA-seq preprocessing, following the scanpy clustering tutorial:
https://scanpy.scverse.org/en/stable/tutorials/basics/clustering.html

    load -> QC metrics -> filter -> doublets -> normalize -> HVG -> PCA -> neighbors/UMAP
         -> Leiden (several resolutions) -> marker-based cell-type annotation

Developer : Abdullah Al Mamun
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# Marker genes from the tutorial (human bone marrow). Replace for other tissues.
MARKER_GENES: dict[str, list[str]] = {
    "CD14+ Mono": ["FCN1", "CD14"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
    "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA"],
    "NK": ["GNLY", "NKG7", "CD247", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    "B cells": ["MS4A1", "ITGB1", "COL4A4", "PRDM1", "IRF4", "PAX5", "BCL11A", "BLK", "IGHD", "IGHM"],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    "Plasmablast": ["XBP1", "PRDM1", "PAX5"],
    "CD4+ T": ["CD4", "IL7R", "TRBC2"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH"],
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
}


@dataclass
class PreprocessConfig:
    min_genes: int = 100                 # tutorial
    min_cells: int = 3                   # tutorial
    max_pct_mt: float | None = None      # e.g. 20; None = no mito filter (tutorial)
    remove_doublets: bool = True         # tutorial only flags them
    n_top_genes: int = 2000              # tutorial
    resolutions: tuple[float, ...] = (0.02, 0.5, 2.0)  # tutorial
    annotate_resolution: float = 0.5     # clusters that get a cell-type label
    marker_genes: dict[str, list[str]] = field(default_factory=lambda: dict(MARKER_GENES))
    seed: int = 0


def res_key(res: float) -> str:
    return f"leiden_res_{res:4.2f}"


def sample_id(path: str | Path) -> str:
    """s1d1_filtered_feature_bc_matrix.h5 -> s1d1"""
    stem = Path(path).name
    stem = re.sub(r"\.(h5|h5ad)$", "", stem)
    return re.sub(r"_?(filtered|raw)_feature_bc_matrix$", "", stem) or stem


def load_samples(paths: list[str | Path]):
    import anndata as ad
    import scanpy as sc

    adatas = {}
    for p in paths:
        a = sc.read_h5ad(p) if str(p).endswith(".h5ad") else sc.read_10x_h5(p)
        a.var_names_make_unique()
        adatas[sample_id(p)] = a
    adata = ad.concat(adatas, label="sample")
    adata.obs_names_make_unique()
    return adata


def annotate_clusters(adata, groupby: str, marker_genes: dict[str, list[str]],
                      key_added: str = "cell_type", min_margin: float = 0.1) -> pd.DataFrame:
    """Label each cluster with the marker set it scores highest on.

    Mirrors reading the tutorial's dotplot (standard_scale="var"): each marker's mean
    log-expression per cluster is min-max scaled across clusters, and a cell type's score
    is the mean over its markers. Scaling per gene keeps very abundant ambient transcripts
    (e.g. HBB in bone marrow) from winning in every cluster.
    """
    import scipy.sparse as sp

    present = {ct: [g for g in genes if g in adata.var_names] for ct, genes in marker_genes.items()}
    missing = [ct for ct, genes in present.items() if not genes]
    present = {ct: genes for ct, genes in present.items() if genes}
    genes = sorted({g for gs in present.values() for g in gs})
    X = adata[:, genes].X
    expr = pd.DataFrame(X.toarray() if sp.issparse(X) else np.asarray(X), columns=genes, index=adata.obs_names)
    labels = adata.obs[groupby].astype(str).to_numpy()
    cluster_mean = expr.groupby(labels).mean()
    cluster_mean = cluster_mean.loc[sorted(cluster_mean.index, key=_natural)]
    span = (cluster_mean.max() - cluster_mean.min()).replace(0, np.nan)
    scaled = ((cluster_mean - cluster_mean.min()) / span).fillna(0)
    means = pd.DataFrame({ct: scaled[gs].mean(axis=1) for ct, gs in present.items()})

    ranked = np.argsort(-means.to_numpy(), axis=1)
    best = means.columns[ranked[:, 0]]
    second = means.columns[ranked[:, 1]] if means.shape[1] > 1 else best
    rows = np.arange(len(means))
    top = means.to_numpy()[rows, ranked[:, 0]]
    runner = means.to_numpy()[rows, ranked[:, 1]] if means.shape[1] > 1 else top
    table = pd.DataFrame({
        "cluster": means.index.astype(str),
        "n_cells": pd.Series(labels).value_counts().reindex(means.index).to_numpy(),
        "cell_type": best,
        "score": top.round(3),
        "runner_up": second,
        "margin": (top - runner).round(3),
        "uncertain": (top - runner) < min_margin,  # check these on the dotplot
    })
    if "pct_counts_mt" in adata.obs:  # high-mito clusters are often damaged cells, not a cell type
        table["mean_pct_mt"] = (adata.obs["pct_counts_mt"].groupby(labels).mean()
                                .reindex(means.index).round(1).to_numpy())
    adata.obs[key_added] = adata.obs[groupby].astype(str).map(dict(zip(table.cluster, table.cell_type)))
    adata.obs[key_added] = adata.obs[key_added].astype("category")
    adata.uns["annotation"] = {"groupby": groupby, "missing_marker_sets": missing}
    return table


def _natural(label: str):
    return (0, int(label), "") if label.isdigit() else (1, 0, label)


def run_pipeline(paths: list[str | Path], cfg: PreprocessConfig | None = None) -> Iterator[tuple[str, dict, object]]:
    import scanpy as sc

    cfg = cfg or PreprocessConfig()
    shape = lambda a: {"cells": a.n_obs, "genes": a.n_vars}

    adata = load_samples(paths)
    yield "load", {**shape(adata), "samples": adata.obs["sample"].value_counts().to_dict()}, adata

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    yield "qc_metrics", {"median_genes": float(adata.obs.n_genes_by_counts.median()),
                         "median_counts": float(adata.obs.total_counts.median()),
                         "median_pct_mt": round(float(adata.obs.pct_counts_mt.median()), 2)}, adata

    before = shape(adata)
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)
    if cfg.max_pct_mt is not None:
        adata = adata[adata.obs.pct_counts_mt < cfg.max_pct_mt].copy()
    yield "filter", {"before": before, "after": shape(adata)}, adata

    sc.pp.scrublet(adata, batch_key="sample", random_state=cfg.seed)
    n_dbl = int(adata.obs.predicted_doublet.sum())
    summary = {"predicted_doublets": n_dbl, "rate": round(n_dbl / adata.n_obs, 4),
               "by_sample": adata.obs.groupby("sample", observed=True).predicted_doublet.sum().astype(int).to_dict()}
    if cfg.remove_doublets:
        adata = adata[~adata.obs.predicted_doublet.astype(bool)].copy()
        summary["after"] = shape(adata)
    yield "doublets", summary, adata

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    yield "normalize", {"layers": [k for k in adata.layers if k]}, adata

    multi = adata.obs["sample"].nunique() > 1
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_genes, batch_key="sample" if multi else None)
    yield "hvg", {"highly_variable": int(adata.var.highly_variable.sum())}, adata

    sc.tl.pca(adata, random_state=cfg.seed)
    sc.pp.neighbors(adata, random_state=cfg.seed)
    sc.tl.umap(adata, random_state=cfg.seed)
    yield "embedding", {"pcs": adata.obsm["X_pca"].shape[1]}, adata

    resolutions = sorted({*cfg.resolutions, cfg.annotate_resolution})
    for res in resolutions:
        sc.tl.leiden(adata, key_added=res_key(res), resolution=res, flavor="igraph",
                     n_iterations=2, random_state=cfg.seed)
    yield "cluster", {res_key(r): int(adata.obs[res_key(r)].nunique()) for r in resolutions}, adata

    groupby = res_key(cfg.annotate_resolution)
    table = annotate_clusters(adata, groupby, cfg.marker_genes)
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")
    top = {c: sc.get.rank_genes_groups_df(adata, group=c).head(5)["names"].tolist() for c in table.cluster}
    table["top_genes"] = table.cluster.map(lambda c: ", ".join(top[c]))
    adata.uns["annotation"]["table"] = table.to_dict("list")
    yield "annotate", {"table": table,
                       "cell_types": adata.obs.cell_type.value_counts().to_dict(),
                       "missing_marker_sets": adata.uns["annotation"]["missing_marker_sets"]}, adata


def summarize_dataset(adata, steps: dict, cfg: PreprocessConfig | None = None) -> dict:
    """Plain facts about a processed dataset, from the AnnData and the per-step summaries
    that run_pipeline yielded. Descriptive only - no statistical tests."""
    cfg = cfg or PreprocessConfig()
    obs = adata.obs
    samples = sorted(obs["sample"].astype(str).unique())
    loaded = steps.get("load", {}).get("samples", {})
    dbl = steps.get("doublets", {}).get("by_sample", {})
    per_sample = []
    for smp in samples:
        o = obs[obs["sample"].astype(str) == smp]
        per_sample.append({"sample": smp, "cells_loaded": int(loaded.get(smp, 0)), "cells_final": int(len(o)),
                           "doublets": int(dbl.get(smp, 0)),
                           "median_genes": int(o["n_genes_by_counts"].median()),
                           "median_counts": int(o["total_counts"].median()),
                           "median_pct_mt": round(float(o["pct_counts_mt"].median()), 1)})

    counts = pd.crosstab(obs["cell_type"].astype(str), obs["sample"].astype(str))
    pct = (counts / counts.sum() * 100).round(1)
    overall = (obs["cell_type"].value_counts(normalize=True) * 100).round(1)
    composition = [{"cell_type": ct, "overall_pct": float(overall[ct]),
                    **{f"{smp}_pct": float(pct.loc[ct, smp]) for smp in samples}}
                   for ct in overall.index]

    diffs = []
    if len(samples) >= 2:
        spread = (pct.max(axis=1) - pct.min(axis=1)).sort_values(ascending=False)
        for ct in spread.index[:5]:
            if spread[ct] >= 1:
                diffs.append({"cell_type": ct, "highest_in": pct.loc[ct].idxmax(), "lowest_in": pct.loc[ct].idxmin(),
                              "range_pct_points": round(float(spread[ct]), 1)})

    table = pd.DataFrame(adata.uns.get("annotation", {}).get("table", {}))
    markers, flagged = {}, []
    if not table.empty:
        for ct, grp in table.sort_values("n_cells", ascending=False).groupby("cell_type", sort=False):
            genes = [g for row in grp["top_genes"] for g in str(row).split(", ") if g]
            markers[ct] = list(dict.fromkeys(genes))[:8]
        for _, r in table.iterrows():
            why = []
            if bool(r.get("uncertain")):
                why.append(f"ambiguous: {r['cell_type']} vs {r['runner_up']} (margin {float(r['margin']):.2f})")
            if r.get("mean_pct_mt", 0) and float(r["mean_pct_mt"]) > 20:
                why.append(f"high mitochondrial reads ({float(r['mean_pct_mt']):.1f}%), likely damaged cells")
            if why:
                flagged.append({"cluster": str(r["cluster"]), "label": r["cell_type"], "n_cells": int(r["n_cells"]),
                                "issues": "; ".join(why)})

    caveats = ["Cell types were assigned automatically from marker genes; confirm them with the dotplot."]
    if len(samples) < 3:
        caveats.append(f"Only {len(samples)} sample(s): differences between samples are descriptive "
                       "and cannot be tested statistically.")
    if "condition" not in obs:
        caveats.append("No condition (e.g. disease/control) is recorded, so nothing here says anything "
                       "about disease.")
    if cfg.max_pct_mt is None:
        caveats.append("No mitochondrial-% filter was applied.")

    return {
        "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars), "samples": per_sample,
        "doublet_rate": steps.get("doublets", {}).get("rate"),
        "n_clusters": int(obs[res_key(cfg.annotate_resolution)].nunique()),
        "annotation_resolution": cfg.annotate_resolution,
        "n_cell_types": int(obs["cell_type"].nunique()), "composition": composition,
        "largest_sample_differences": diffs, "top_genes_by_cell_type": markers,
        "flagged_clusters": flagged, "caveats": caveats,
        "conditions": obs.groupby("sample", observed=True)["condition"].first().astype(str).to_dict()
        if "condition" in obs else None,
    }


def summary_markdown(f: dict, interpretation: str | None = None) -> str:
    """The facts from summarize_dataset as a readable report."""
    lines = ["# Dataset summary", "",
             f"**{f['n_cells']:,} cells** and **{f['n_genes']:,} genes** from **{len(f['samples'])} samples**, "
             f"grouped into {f['n_clusters']} clusters (Leiden resolution {f['annotation_resolution']}) and "
             f"{f['n_cell_types']} cell types.", ""]
    if interpretation:
        lines += ["## Interpretation", "", interpretation.strip(), ""]
    lines += ["## Samples and quality", "",
              "| Sample | Loaded | Final | Doublets | Median genes | Median UMIs | Median % mito |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {s['sample']} | {s['cells_loaded']:,} | {s['cells_final']:,} | {s['doublets']} | "
              f"{s['median_genes']:,} | {s['median_counts']:,} | {s['median_pct_mt']} |" for s in f["samples"]]
    smp = [s["sample"] for s in f["samples"]]
    lines += ["", "## Cell-type composition (% of cells)", "",
              "| Cell type | All | " + " | ".join(smp) + " | Top genes |", "|---" * (len(smp) + 3) + "|"]
    for c in f["composition"]:
        genes = ", ".join(f["top_genes_by_cell_type"].get(c["cell_type"], [])[:5])
        lines.append(f"| {c['cell_type']} | {c['overall_pct']} | "
                     + " | ".join(str(c[f"{s}_pct"]) for s in smp) + f" | {genes} |")
    if f["largest_sample_differences"]:
        lines += ["", "## Largest differences between samples (descriptive)", ""]
        lines += [f"- **{d['cell_type']}**: {d['range_pct_points']} percentage points higher in "
                  f"{d['highest_in']} than {d['lowest_in']}" for d in f["largest_sample_differences"]]
    if f["flagged_clusters"]:
        lines += ["", "## Clusters to check", ""]
        lines += [f"- Cluster {c['cluster']} ({c['label']}, {c['n_cells']:,} cells): {c['issues']}"
                  for c in f["flagged_clusters"]]
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in f["caveats"]]
    return "\n".join(lines) + "\n"


INTERPRETATION_PROMPT = """You are an experienced single-cell RNA-seq analyst. Below are facts computed \
from a processed dataset (JSON). Write a short interpretation for a biologist, in Markdown, with \
these parts: what tissue/cell populations the data appears to contain; the dominant and rare \
populations; notable differences between samples (descriptive only); data-quality concerns; \
and 3 concrete next steps.

Rules: use only the facts given; do not invent numbers, p-values or genes; sample names are \
opaque IDs, so do not read meaning (timepoints, donors, treatments) into them; say plainly when \
something cannot be concluded (e.g. too few samples, no disease labels); keep it under 300 words.

Facts:
{facts}"""


def save_figures(adata, outdir: str | Path, cfg: PreprocessConfig | None = None) -> list[Path]:
    """The tutorial's main plots, written as PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc

    cfg = cfg or PreprocessConfig()
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    groupby = res_key(cfg.annotate_resolution)
    markers = {ct: [g for g in genes if g in adata.var_names] for ct, genes in cfg.marker_genes.items()}
    markers = {ct: g for ct, g in markers.items() if g}
    plots = {
        "qc_violin": lambda: sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                                          jitter=0.4, multi_panel=True, show=False),
        "umap_sample_doublets": lambda: sc.pl.umap(adata, color=["sample", "doublet_score"], wspace=0.5,
                                                   size=3, show=False),
        "umap_clusters": lambda: sc.pl.umap(adata, color=[res_key(r) for r in cfg.resolutions],
                                            legend_loc="on data", show=False),
        "umap_cell_type": lambda: sc.pl.umap(adata, color=["cell_type", groupby], wspace=0.6, show=False),
        "dotplot_markers": lambda: sc.pl.dotplot(adata, markers, groupby=groupby, standard_scale="var",
                                                 show=False),
    }
    written = []
    for name, draw in plots.items():
        draw()
        path = out / f"{name}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close("all")
        written.append(path)
    return written


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Automated scanpy preprocessing (clustering tutorial).")
    ap.add_argument("inputs", nargs="+", help="10x .h5 (or .h5ad) files, one per sample")
    ap.add_argument("--out", default="dataset/processed.h5ad")
    ap.add_argument("--figures", default="runs/preprocess_figures")
    ap.add_argument("--max-pct-mt", type=float)
    ap.add_argument("--keep-doublets", action="store_true")
    ap.add_argument("--resolution", type=float, default=0.5, help="Leiden resolution to annotate")
    args = ap.parse_args()

    cfg = PreprocessConfig(max_pct_mt=args.max_pct_mt, remove_doublets=not args.keep_doublets,
                           annotate_resolution=args.resolution)
    adata = None
    for step, summary, adata in run_pipeline(args.inputs, cfg):
        if step == "annotate":
            print(f"[{step}]")
            print(summary["table"].to_string(index=False))
        else:
            print(f"[{step}] {summary}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print("wrote", args.out)
    for p in save_figures(adata, args.figures, cfg):
        print("figure", p)


if __name__ == "__main__":
    main()
