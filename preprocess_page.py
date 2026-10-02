"""Streamlit page: automated scanpy preprocessing (QC, doublets, normalization,
clustering, cell-type annotation) with a review step before saving. Used by app.py."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agentic_ml.preprocess import (INTERPRETATION_PROMPT, MARKER_GENES, PreprocessConfig, res_key, run_pipeline,
                                   sample_id, summarize_dataset, summary_markdown)
from agentic_ml.providers import PRESETS, make_chat, resolve_key
from llm_settings import ALLOW_ENV_KEYS, agent_settings

APP_DIR = Path(__file__).parent
DATASET_DIR = APP_DIR / "dataset"
STEP_LABELS = {"load": "Load samples", "qc_metrics": "QC metrics", "filter": "Filter cells & genes",
               "doublets": "Doublet detection (Scrublet)", "normalize": "Normalize + log1p",
               "hvg": "Highly variable genes", "embedding": "PCA, neighbors, UMAP",
               "cluster": "Leiden clustering", "annotate": "Cell-type annotation"}
EXTRA_LABELS = ["Low quality", "Unassigned"]


def _find_inputs(folder: Path) -> list[Path]:
    files = [*folder.rglob("*.h5"), *folder.rglob("*.h5ad")]
    return sorted(p for p in files if "processed" not in p.name)


def _show(draw):
    """Render a scanpy plot (called with show=False) into Streamlit."""
    draw()
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close("all")


def _settings() -> tuple[list[Path], PreprocessConfig, Path]:
    st.subheader("Input")
    folder = Path(st.text_input("Dataset folder", str(DATASET_DIR)))
    found = _find_inputs(folder) if folder.exists() else []
    if not found:
        st.warning(f"No .h5 / .h5ad files under {folder}")
    picked = st.multiselect("Samples", found, default=found, format_func=lambda p: f"{sample_id(p)}  ({p.name})")

    st.subheader("QC")
    min_genes = st.number_input("Min genes per cell", 0, 5000, 100)
    min_cells = st.number_input("Min cells per gene", 0, 1000, 3)
    mt_on = st.checkbox("Filter on % mitochondrial", False, help="Not in the tutorial; damaged cells have high mito %.")
    max_mt = st.slider("Max % mitochondrial", 5, 50, 20) if mt_on else None
    remove_dbl = st.toggle("Remove predicted doublets", True, help="The tutorial only flags them.")

    st.subheader("Clustering")
    n_top = st.number_input("Highly variable genes", 500, 10000, 2000, step=500)
    res = st.select_slider("Annotate at Leiden resolution", [0.02, 0.1, 0.25, 0.5, 1.0, 2.0], value=0.5)
    with st.expander("Marker genes (JSON)"):
        st.caption("Tutorial markers for human bone marrow. Edit for other tissues.")
        markers_txt = st.text_area("markers", json.dumps(MARKER_GENES, indent=1), height=260,
                                   label_visibility="collapsed")
    try:
        markers = json.loads(markers_txt)
    except json.JSONDecodeError as e:
        st.error(f"Marker JSON is invalid: {e}")
        markers = MARKER_GENES

    out = Path(st.text_input("Save to", str(folder / "processed.h5ad")))
    cfg = PreprocessConfig(min_genes=int(min_genes), min_cells=int(min_cells), max_pct_mt=max_mt,
                           remove_doublets=remove_dbl, n_top_genes=int(n_top),
                           annotate_resolution=res, marker_genes=markers)
    return picked, cfg, out


def _run(paths, cfg):
    log = st.container()
    adata, summaries = None, {}
    for step, summary, adata in run_pipeline(paths, cfg):
        summaries[step] = summary
        with log.status(f"✓ {STEP_LABELS[step]}", state="complete", expanded=False):
            if step == "annotate":
                st.dataframe(summary["table"], hide_index=True)
            else:
                st.json(summary)
    return adata, summaries


def _results(adata, cfg: PreprocessConfig, out: Path):
    import scanpy as sc

    groupby = res_key(cfg.annotate_resolution)
    markers = {ct: [g for g in gs if g in adata.var_names] for ct, gs in cfg.marker_genes.items()}
    markers = {ct: gs for ct, gs in markers.items() if gs}
    qc_tab, umap_tab, ann_tab, sum_tab, save_tab = st.tabs(["QC", "UMAP", "Annotation", "Summary", "Save"])

    with qc_tab:
        _show(lambda: sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                                   jitter=0.4, multi_panel=True, show=False))
        _show(lambda: sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", show=False))
    with umap_tab:
        color = st.multiselect("Color by", ["cell_type", groupby, "sample", "doublet_score", "pct_counts_mt",
                                            "log1p_total_counts", *[res_key(r) for r in cfg.resolutions]],
                               default=["cell_type", "sample"])
        if color:
            _show(lambda: sc.pl.umap(adata, color=color, wspace=0.6, ncols=2, size=3, show=False))
    with ann_tab:
        st.caption("Automatic labels from marker genes. Review the dotplot and the **uncertain** / high "
                   "**mean_pct_mt** rows, correct `cell_type` where needed, then apply.")
        table = pd.DataFrame(adata.uns["annotation"]["table"])
        options = sorted({*markers, *EXTRA_LABELS, *table.cell_type})
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key="ann_editor",
            disabled=[c for c in table.columns if c != "cell_type"],
            column_config={"cell_type": st.column_config.SelectboxColumn(options=options, required=True),
                           "uncertain": st.column_config.CheckboxColumn()})
        if st.button("Apply labels"):
            mapping = dict(zip(edited.cluster.astype(str), edited.cell_type))
            adata.obs["cell_type"] = adata.obs[groupby].astype(str).map(mapping).astype("category")
            adata.uns["annotation"]["table"] = edited.to_dict("list")
            st.success("Labels updated.")
        _show(lambda: sc.pl.dotplot(adata, markers, groupby=groupby, standard_scale="var", show=False))
    with sum_tab:
        _summary(adata, cfg)
    with save_tab:
        samples = sorted(adata.obs["sample"].unique())
        st.caption("Optional: a condition per sample (e.g. disease / control), needed later for the "
                   "biomarker search. It needs at least 4 samples per condition.")
        cond = st.data_editor(pd.DataFrame({"sample": samples, "condition": [""] * len(samples)}),
                              hide_index=True, disabled=["sample"], key="cond_editor")
        drop_lq = st.checkbox("Drop clusters labelled 'Low quality'", True)
        if st.button("💾 Save .h5ad", type="primary"):
            to_save = adata
            if drop_lq and (adata.obs["cell_type"] == "Low quality").any():
                to_save = adata[adata.obs["cell_type"] != "Low quality"].copy()
            conds = dict(zip(cond["sample"], cond["condition"].fillna("").str.strip()))
            if any(conds.values()):
                to_save.obs["condition"] = to_save.obs["sample"].astype(str).map(conds).astype("category")
            to_save.uns["preprocess_config"] = json.dumps(asdict(cfg), default=str)
            out.parent.mkdir(parents=True, exist_ok=True)
            to_save.write_h5ad(out)
            st.success(f"Saved {to_save.n_obs:,} cells × {to_save.n_vars:,} genes to {out}")


def _summary(adata, cfg: PreprocessConfig):
    facts = summarize_dataset(adata, st.session_state.pp_summaries, cfg)
    st.caption("Computed from the current labels. Apply label edits on the Annotation tab first.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells", f"{facts['n_cells']:,}")
    c2.metric("Samples", len(facts["samples"]))
    c3.metric("Cell types", facts["n_cell_types"])
    c4.metric("Clusters to check", len(facts["flagged_clusters"]))

    st.markdown("**Cell-type composition (% of cells)**")
    comp = pd.DataFrame(facts["composition"])
    long = comp.melt(id_vars="cell_type", value_vars=[c for c in comp if c.endswith("_pct") and c != "overall_pct"],
                     var_name="sample", value_name="pct")
    long["sample"] = long["sample"].str.removesuffix("_pct")
    st.bar_chart(long, x="sample", y="pct", color="cell_type", horizontal=True, height=90 + 40 * len(facts["samples"]))
    comp["top_genes"] = comp.cell_type.map(lambda c: ", ".join(facts["top_genes_by_cell_type"].get(c, [])[:5]))
    st.dataframe(comp, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Largest differences between samples** (descriptive)")
        for d in facts["largest_sample_differences"] or [{"cell_type": None}]:
            st.markdown(f"- **{d['cell_type']}**: {d['range_pct_points']} pp higher in {d['highest_in']} "
                        f"than {d['lowest_in']}" if d["cell_type"] else "- none above 1 percentage point")
        st.markdown("**Caveats**")
        st.markdown("\n".join(f"- {c}" for c in facts["caveats"]))
    with right:
        st.markdown("**Clusters to check**")
        if facts["flagged_clusters"]:
            st.dataframe(pd.DataFrame(facts["flagged_clusters"]), hide_index=True, width="stretch")
        else:
            st.markdown("None flagged.")

    st.divider()
    st.markdown("**Written interpretation** (optional): a model reads only the facts above.")
    with st.expander("Model", expanded="pp_interpretation" not in st.session_state):
        llm = agent_settings()
    if st.button("✍️ Write interpretation"):
        key = resolve_key(llm["provider"], llm["api_key"], allow_env=ALLOW_ENV_KEYS)
        if PRESETS[llm["provider"]].needs_key and not key:
            st.error(f"Enter your {PRESETS[llm['provider']].label} API key.")
        elif not llm["model"]:
            st.error("Choose a model.")
        else:
            try:
                with st.spinner(f"Asking {llm['model']}…"):
                    chat = make_chat(llm["provider"], llm["model"], "You write concise, accurate scientific summaries.",
                                     [], api_key=key, base_url=llm["base_url"])
                    chat.user(INTERPRETATION_PROMPT.format(facts=json.dumps(facts, indent=1, default=str)))
                    text = "\n\n".join(chat.step().texts).strip()
                st.session_state.pp_interpretation = f"{text}\n\n*Written by {llm['model']} from the computed facts.*"
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    interp = st.session_state.get("pp_interpretation")
    if interp:
        with st.container(border=True):
            st.markdown(interp)

    st.download_button("⬇️ Download summary (.md)", summary_markdown(facts, interp),
                       file_name="dataset_summary.md", mime="text/markdown")


def preprocess_page():
    with st.sidebar:
        paths, cfg, out = _settings()
        start = st.button("▶ Run preprocessing", type="primary", width="stretch", disabled=not paths)

    st.title("🧫 Preprocessing")
    st.caption("Automates the scanpy clustering tutorial: QC → doublets → normalization → HVG → PCA/UMAP → "
               "Leiden → marker-based cell-type annotation. Takes about a minute for 17k cells.")
    if start:
        with st.spinner("Running pipeline…"):
            adata, summaries = _run(paths, cfg)
        st.session_state.update(pp_adata=adata, pp_cfg=cfg, pp_summaries=summaries)
        st.session_state.pop("pp_interpretation", None)  # belongs to the previous run
        st.rerun()

    if "pp_adata" not in st.session_state:
        st.info("Pick samples in the sidebar and run.")
        return
    adata, run_cfg = st.session_state.pp_adata, st.session_state.pp_cfg
    s = st.session_state.pp_summaries
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells loaded", f"{s['load']['cells']:,}")
    c2.metric("After QC", f"{s['filter']['after']['cells']:,}")
    c3.metric("Doublets", f"{s['doublets']['predicted_doublets']:,}", help=f"rate {s['doublets']['rate']:.1%}")
    c4.metric("Clusters", s["cluster"][res_key(run_cfg.annotate_resolution)])
    _results(adata, run_cfg, out)
