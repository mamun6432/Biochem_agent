"""Streamlit dashboard for the agentic model search.

    pip install -r requirements.txt
    streamlit run app.py



Three modes:
  * Preprocess - automated scanpy QC, doublets, normalization, clustering and
                 cell-type annotation, with a review step before saving an .h5ad.
  * Run agent  - start a search and watch each proposed experiment (and what it
                 changes vs. the previous one) before and after the harness scores it.
  * Browse ledger - inspect any runs/*.jsonl ledger, optionally auto-refreshing so
                 you can follow a `python run_agent.py` run from the terminal.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agentic_ml import (AgentTools, ExperimentSpec, Harness, Ledger, SYSTEM_PROMPT, TOOL_DEFINITIONS,
                        build_sample_dataset, from_anndata)
from agentic_ml.agent import agent_events, spec_from_call
from agentic_ml.providers import PRESETS, make_chat, resolve_key
from llm_settings import ALLOW_ENV_KEYS, agent_settings
from preprocess_page import preprocess_page

APP_DIR = Path(__file__).parent
RUNS_DIR = APP_DIR / "runs"
STATUS_COLORS = {"ok": "#2a9d8f", "no_signal": "#e9a23b", "rejected": "#e76f51"}
DIFF_IGNORE = {"name", "rationale"}

st.set_page_config(page_title="Agentic biomarker search", page_icon="🧬", layout="wide")


# ----------------------------------------------------------------- helpers
def normalize_spec(call_input: dict) -> dict:
    """The proposed spec with ExperimentSpec defaults filled in, so diffs compare like with like."""
    try:
        spec = spec_from_call(call_input)
    except ValueError:  # malformed; the tool result will show the error
        return {}
    try:
        return json.loads(ExperimentSpec.from_dict(spec).to_json())
    except TypeError:  # unknown keys - the harness will reject it anyway
        return spec


def fmt(v) -> str:
    if v is None:
        return "all"
    if isinstance(v, list):
        return ", ".join(map(str, v)) if v else "[]"
    if isinstance(v, dict):
        return json.dumps(v) if v else "{}"
    return str(v)


def spec_diff(prev: dict | None, new: dict) -> list[tuple[str, str, str]]:
    if prev is None:
        return []
    keys = [k for k in new if k not in DIFF_IGNORE]
    return [(k, fmt(prev.get(k)), fmt(new.get(k))) for k in keys if prev.get(k) != new.get(k)]


def render_diff(prev: dict | None, new: dict):
    if prev is None:
        st.caption("First experiment — baseline spec:")
        st.code("\n".join(f"{k}: {fmt(v)}" for k, v in new.items() if k not in DIFF_IGNORE), language="yaml")
        return
    changes = spec_diff(prev, new)
    if not changes:
        st.caption("No spec changes vs. previous experiment.")
        return
    st.markdown("**Changes vs. previous experiment**")
    st.dataframe(pd.DataFrame(changes, columns=["field", "before", "after"]),
                 hide_index=True, width="stretch")


def result_label(name: str, r: dict) -> tuple[str, str]:
    status = r.get("status")
    if status == "rejected" or "error" in r:
        return f"✗ {name} — rejected: {r.get('message') or r.get('error')}", "error"
    ci = r.get("cv_auc_ci") or [None, None]
    tag = "⚠ no signal" if status == "no_signal" else "✓"
    return (f"{tag} {name} — CV AUC {r['cv_auc']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}] · "
            f"{r.get('n_features')} features · stability {r.get('feature_stability') or 0:.2f}"), "complete"


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # half-written last line while a run is in progress
                pass
    return out


# ---------------------------------------------------------- scoreboard
def render_board(entries: list[dict], budget: int | None = None):
    exps = [(i, e) for i, e in enumerate(entries) if e["result"].get("status") != "finalized"]
    final = next((e for e in reversed(entries) if e["result"].get("status") == "finalized"), None)
    scored = [(i, e) for i, e in exps if e["result"].get("cv_auc") is not None]
    best = max(scored, key=lambda t: t[1]["result"]["cv_auc"], default=None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiments", f"{len(exps)}" + (f" / {budget}" if budget else ""))
    c2.metric("Best CV AUC", f"{best[1]['result']['cv_auc']:.3f}" if best else "—",
              help=f"#{best[0]} {best[1]['spec']['name']}" if best else None)
    c3.metric("Hold-out AUC", f"{final['result']['holdout_auc']:.3f}"
              if final and final["result"].get("holdout_auc") is not None else "sealed")

    if not exps:
        st.info("No experiments yet.")
        return

    rows = []
    for i, e in exps:
        r, s = e["result"], e["spec"]
        ci = r.get("cv_auc_ci") or [None, None]
        rows.append({"id": i, "name": s["name"], "status": r["status"], "model": s["model"],
                     "blocks": fmt(s["feature_blocks"]), "cell_types": fmt(s["cell_types"]),
                     "selection": s["selection"], "n_features": r.get("n_features"),
                     "cv_auc": r.get("cv_auc"), "ci_lo": ci[0], "ci_hi": ci[1],
                     "overfit_gap": r.get("overfit_gap"), "stability": r.get("feature_stability"),
                     "top5": ", ".join(t["feature"] for t in r.get("top_features", [])[:5])})
    df = pd.DataFrame(rows)

    plot = df.dropna(subset=["cv_auc"])
    if not plot.empty:
        lo = max(0.0, min(plot["ci_lo"].min(), 0.5) - 0.05)
        y = alt.Y("cv_auc:Q", title="CV AUC (95% CI)", scale=alt.Scale(domain=[lo, 1.0]))
        color = alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLORS),
                                                      range=list(STATUS_COLORS.values())))
        base = alt.Chart(plot).encode(x=alt.X("id:O", title="Experiment"))
        chart = (base.mark_rule(strokeWidth=2).encode(y=alt.Y("ci_lo:Q", scale=alt.Scale(domain=[lo, 1.0])),
                                                      y2="ci_hi:Q", color=color)
                 + base.mark_circle(size=110, opacity=1).encode(
                     y=y, color=color,
                     tooltip=["id", "name", "model", "n_features", alt.Tooltip("cv_auc:Q", format=".3f"),
                              alt.Tooltip("overfit_gap:Q", format=".3f"),
                              alt.Tooltip("stability:Q", format=".2f"), "top5"])
                 + alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray")
                 .encode(y="y:Q"))
        st.altair_chart(chart.properties(height=260), width="stretch")

    st.dataframe(df.drop(columns=["ci_lo", "ci_hi"]), hide_index=True, width="stretch",
                 column_config={"cv_auc": st.column_config.NumberColumn(format="%.3f"),
                                "overfit_gap": st.column_config.NumberColumn(format="%.3f"),
                                "stability": st.column_config.NumberColumn(format="%.2f")})

    if final:
        r = final["result"]
        ci = r.get("holdout_auc_ci") or [None, None]
        st.success(f"**Finalized experiment #{r.get('finalized_experiment')}** — hold-out AUC "
                   + (f"{r['holdout_auc']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]" if r.get("holdout_auc") is not None
                      else "n/a") + f" on {r.get('n_holdout')} samples")
        if r.get("panel"):
            st.dataframe(pd.DataFrame(r["panel"]), hide_index=True, width="stretch")


def render_details(entries: list[dict]):
    exps = [(i, e) for i, e in enumerate(entries) if e["result"].get("status") != "finalized"]
    if not exps:
        return
    labels = {i: f"#{i} {e['spec']['name']}" for i, e in exps}
    pick = st.selectbox("Inspect experiment", list(labels), index=len(labels) - 1,
                        format_func=labels.get, key="inspect")
    e = entries[pick]
    prev = next((entries[j]["spec"] for j, _ in reversed(exps) if j < pick), None)
    s, r = e["spec"], e["result"]
    st.markdown(f"**Rationale:** {s['rationale']}")
    if e.get("reflection"):
        st.markdown(f"**Reflection on prior results:** {e['reflection']}")
    if r.get("message"):
        st.warning(r["message"])
    left, right = st.columns(2)
    with left:
        render_diff(prev, s)
    with right:
        if r.get("fold_aucs"):
            st.markdown("**Per-fold AUC**")
            st.bar_chart(pd.DataFrame({"fold": range(1, len(r["fold_aucs"]) + 1), "auc": r["fold_aucs"]}),
                         x="fold", y="auc", height=180)
        if r.get("top_features"):
            st.markdown("**Top features**")
            st.dataframe(pd.DataFrame(r["top_features"]).head(20), hide_index=True, width="stretch")
    with st.expander("Raw spec + result JSON"):
        st.json({"spec": s, "result": r})


# ---------------------------------------------------------- live event log
class EventRenderer:
    """Draws agent events into a container; used live and to replay after a rerun."""

    def __init__(self, container):
        self.c = container
        self.prev_spec: dict | None = None
        self.pending = None

    def handle(self, ev: dict):
        with self.c:
            t = ev["type"]
            if t == "text":
                st.chat_message("assistant").markdown(ev["text"])
            elif t == "tool_call":
                self._call(ev)
            elif t == "tool_result":
                self._result(ev)
            elif t == "done":
                st.caption("Agent finished.")
            elif t == "note":
                st.caption(f"ℹ️ {ev['text']}")
            elif t == "error":
                st.error(ev["text"])

    def _call(self, ev):
        name, inp = ev["name"], ev["input"]
        if name == "run_experiment":
            spec = normalize_spec(inp)
            self.pending = st.status(f"🧪 Proposed: **{spec.get('name', '?')}** — running…",
                                     expanded=True, state="running")
            with self.pending:
                st.markdown(f"*{spec.get('rationale', '')}*")
                if inp.get("reflection"):
                    st.caption(f"Reflection: {inp['reflection']}")
                render_diff(self.prev_spec, spec)
            self.prev_spec = spec
        elif name == "finalize":
            self.pending = st.status(f"🔓 Finalizing experiment #{inp.get('experiment_id')} "
                                     "on the sealed hold-out…", state="running")
        else:
            self.pending = st.status(f"🔧 {name}", state="running")

    def _result(self, ev):
        name, out = ev["name"], ev["output"]
        box, self.pending = self.pending, None
        if box is None:
            return
        if name == "run_experiment":
            label, state = result_label(out.get("name") or normalize_spec(ev["input"]).get("name", "?"), out)
            box.update(label=label, state=state, expanded=False)
        elif name == "finalize":
            if "error" in out:
                box.update(label=f"Finalize failed: {out['error']}", state="error")
            else:
                ci = out.get("holdout_auc_ci") or [0, 0]
                box.update(label=f"🔓 Hold-out AUC {out.get('holdout_auc') or float('nan'):.3f} "
                                 f"[{ci[0]:.2f}, {ci[1]:.2f}] on {out.get('n_holdout')} samples",
                           state="complete")
        else:
            with box:
                if name == "get_data_card" and isinstance(out, dict):
                    st.json({k: v for k, v in out.items() if k != "spec_schema"}, expanded=False)
                else:
                    st.json(out, expanded=False)
            box.update(label=f"🔧 {name}", state="error" if isinstance(out, dict) and "error" in out
                       else "complete", expanded=False)


# ------------------------------------------------------------------ pages
def load_dataset(cfg: dict):
    if cfg["source"] == "h5ad":
        import anndata
        return from_anndata(anndata.read_h5ad(cfg["h5ad"]), cfg["sample"], cfg["label"],
                            cfg["celltype"], cfg["positive"])
    from agentic_ml.simulate import simulate_cells
    counts, obs, genes = simulate_cells(n_samples=cfg["n_samples"], effect=cfg["effect"])
    return build_sample_dataset(counts, obs, genes, "sample", "condition", "cell_type", "disease")


def run_page():
    with st.sidebar:
        st.subheader("Data")
        source = st.radio("Source", ["simulated", "h5ad"], horizontal=True)
        cfg = {"source": source}
        if source == "simulated":
            cfg["n_samples"] = st.slider("Samples", 16, 96, 32, step=4)
            cfg["effect"] = st.slider("Planted effect size", 0.0, 4.0, 2.0, step=0.25,
                                      help="0 = no real signal; the agent should conclude no_signal")
        else:
            cfg["h5ad"] = st.text_input("Path to .h5ad")
            cfg["sample"] = st.text_input("Sample column", "sample_id")
            cfg["label"] = st.text_input("Label column", "condition")
            cfg["celltype"] = st.text_input("Cell-type column", "cell_type")
            cfg["positive"] = st.text_input("Positive label", "disease")

        llm = agent_settings()

        st.subheader("Search")
        budget = st.number_input("Experiment budget", 1, 100, 12)
        ctx_file = st.file_uploader("DE / enrichment context (.txt, .md, .csv)", type=["txt", "md", "csv", "tsv"])
        ctx_text = st.text_area("…or paste context", height=100)
        # default is fixed once per session so the widget keeps a stable identity across reruns
        st.session_state.setdefault("ledger_path", str(RUNS_DIR / f"ledger_{datetime.now():%Y%m%d_%H%M%S}.jsonl"))
        ledger = st.text_input("Ledger file", key="ledger_path",
                               help="Use a new file per run: an existing ledger's entries are loaded "
                                    "and the agent will see them.")
        start = st.button("▶ Start search", type="primary", width="stretch",
                          disabled=st.session_state.get("running", False))

    st.title("🧬 Agentic biomarker search")
    st.caption("Watch each experiment the agent proposes — its rationale and exactly what it changes — "
               "before the harness scores it. Changing sidebar widgets mid-run will stop the run.")

    log_col, board_col = st.columns([5, 6], gap="large")
    with log_col:
        st.subheader("Agent log")
        log = st.container(height=750)
    with board_col:
        st.subheader("Scoreboard")
        board = st.empty()

    if start:
        llm["api_key"] = resolve_key(llm["provider"], llm["api_key"], allow_env=ALLOW_ENV_KEYS)
        if PRESETS[llm["provider"]].needs_key and not llm["api_key"]:
            st.sidebar.error(f"Enter your {PRESETS[llm['provider']].label} API key.")
            return
        if not llm["model"]:
            st.sidebar.error("Choose a model.")
            return
        context = (ctx_file.getvalue().decode(errors="replace") if ctx_file else ctx_text.strip()) \
            or "(no DE/enrichment context provided)"
        st.session_state.update(events=[], ledger=ledger, budget=int(budget), running=True,
                                run_model=f"{PRESETS[llm['provider']].label} · {llm['model']}")
        _run(cfg, llm, int(budget), ledger, context, log, board)
        st.session_state.running = False
        st.session_state.pop("ledger_path", None)  # next run gets a fresh ledger file
        st.rerun()  # redraw with the details panel and re-enable the button

    # replay the last run after any rerun
    if "run_model" in st.session_state:
        log_col.caption(f"Last run: {st.session_state.run_model}")
    renderer = EventRenderer(log)
    for ev in st.session_state.get("events", []):
        renderer.handle(ev)
    entries = load_ledger(Path(st.session_state["ledger"])) if "ledger" in st.session_state else []
    with board.container():
        render_board(entries, st.session_state.get("budget"))
    if entries:
        st.divider()
        render_details(entries)


def _run(cfg, llm, budget, ledger, context, log, board):
    events = st.session_state.events
    renderer = EventRenderer(log)

    def emit(ev):
        events.append(ev)
        renderer.handle(ev)

    try:
        with log, st.spinner("Building sample-level dataset…"):
            tools = AgentTools(Harness(load_dataset(cfg), budget=budget), Ledger(ledger))
    except Exception as e:
        emit({"type": "error", "text": f"Could not load data: {e}"})
        return
    with board.container():
        render_board(tools.ledger.entries, budget)

    try:
        chat = make_chat(llm["provider"], llm["model"], SYSTEM_PROMPT, TOOL_DEFINITIONS,
                         api_key=llm["api_key"], base_url=llm["base_url"])
        for ev in agent_events(chat, tools, context):
            emit(ev)
            if ev["type"] == "tool_result" and ev["name"] in ("run_experiment", "finalize"):
                with board.container():
                    render_board(tools.ledger.entries, budget)
    except Exception as e:
        hint = (" — is the server running and the model pulled?"
                if llm["provider"] in ("ollama", "custom") and "connect" in str(e).lower() else "")
        emit({"type": "error", "text": f"{type(e).__name__}: {e}{hint}"})


def browse_page():
    files = sorted({*RUNS_DIR.glob("*.jsonl"), *Path("runs").resolve().glob("*.jsonl")},
                   key=lambda p: p.stat().st_mtime, reverse=True)
    with st.sidebar:
        st.subheader("Ledger")
        pick = st.selectbox("Ledger file", files, format_func=lambda p: p.name) if files else None
        custom = st.text_input("…or path", placeholder="runs/ledger.jsonl")
        live = st.toggle("Auto-refresh every 3 s", help="Follow a run started with `python run_agent.py`")
    path = Path(custom) if custom else pick

    st.title("📒 Ledger browser")
    if path is None:
        st.info(f"No ledgers found in {RUNS_DIR}. Run a search first or enter a path.")
        return
    st.caption(str(path))

    @st.fragment(run_every=3 if live else None)
    def view():
        entries = load_ledger(path)
        render_board(entries)
        if entries:
            st.divider()
            render_details(entries)

    view()


PAGES = {"Preprocess": preprocess_page, "Run agent": run_page, "Browse ledger": browse_page}
page = st.sidebar.radio("Mode", list(PAGES), horizontal=True)
st.sidebar.divider()
PAGES[page]()
