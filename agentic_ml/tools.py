"""The four tools exposed to the LLM agent, plus the persistent ledger.

Wire these into whatever agent framework you use (Claude Agent SDK, LangGraph,
plain tool-use loop). Each returns JSON-serialisable dicts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .harness import Harness
from .spec import ExperimentSpec, SPEC_JSON_SCHEMA


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        if self.path.exists():
            self.entries = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def append(self, spec: ExperimentSpec, result: dict, reflection: str = ""):
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "spec": json.loads(spec.to_json()),
                 "result": result, "reflection": reflection}
        self.entries.append(entry)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def summary(self) -> list[dict]:
        """Compact view for the agent: one line per experiment."""
        out = []
        for i, e in enumerate(self.entries):
            r = e["result"]
            out.append({
                "id": i, "name": e["spec"]["name"], "status": r["status"],
                "model": e["spec"]["model"], "blocks": e["spec"]["feature_blocks"],
                "cell_types": e["spec"]["cell_types"], "n_features": r.get("n_features"),
                "cv_auc": r.get("cv_auc"), "ci": r.get("cv_auc_ci"),
                "overfit_gap": r.get("overfit_gap"), "stability": r.get("feature_stability"),
                "top5": [t["feature"] for t in r.get("top_features", [])[:5]],
                "reflection": e.get("reflection", ""),
            })
        return out


class AgentTools:
    """Bind a Harness + Ledger and expose the tool functions."""

    def __init__(self, harness: Harness, ledger: Ledger):
        self.h = harness
        self.ledger = ledger

    # ---- tool 1
    def get_data_card(self) -> dict:
        card = self.h.data_card()
        card["spec_schema"] = SPEC_JSON_SCHEMA
        return card

    # ---- tool 2
    def run_experiment(self, spec: dict, reflection: str = "") -> dict:
        s = ExperimentSpec.from_dict(spec)
        res = self.h.run_experiment(s).to_dict()
        self.ledger.append(s, res, reflection)
        return res

    # ---- tool 3
    def read_ledger(self) -> list[dict]:
        return self.ledger.summary()

    # ---- tool 4
    def finalize(self, experiment_id: int) -> dict:
        ok = [i for i, e in enumerate(self.ledger.entries) if e["result"].get("status") in ("ok", "no_signal")]
        if experiment_id not in ok:
            raise ValueError(f"experiment_id {experiment_id} is not a scored experiment; "
                             f"choose one of {ok} (see read_ledger)")
        spec = ExperimentSpec.from_dict(self.ledger.entries[experiment_id]["spec"])
        out = self.h.finalize(spec)
        out["finalized_experiment"] = experiment_id
        self.ledger.append(spec, {"status": "finalized", **out}, "FINAL hold-out evaluation")
        return out


# Tool definitions in Anthropic tool-use format, ready to pass to the API.
TOOL_DEFINITIONS = [
    {"name": "get_data_card",
     "description": "Summary of the development set (sample counts, cell types, feature blocks) "
                    "and the JSON schema for experiment specs. Call first.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_experiment",
     "description": "Run one experiment under sample-level cross-validation. Returns CV AUC with CI, "
                    "overfit gap, feature stability and top features. Costs one unit of budget.",
     "input_schema": {"type": "object", "required": ["spec", "reflection"],
                      "properties": {"spec": SPEC_JSON_SCHEMA,
                                     "reflection": {"type": "string",
                                                    "description": "What you expect and why, given prior results."}}}},
    {"name": "read_ledger",
     "description": "All experiments run so far with their key metrics and your reflections.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "finalize",
     "description": "Select one experiment as the final biomarker model. Evaluates it ONCE on the sealed "
                    "hold-out. Can only be called once; ends the search.",
     "input_schema": {"type": "object", "required": ["experiment_id"],
                      "properties": {"experiment_id": {"type": "integer"}}}},
]

SYSTEM_PROMPT = """You are running a bounded model search to find a compact, robust diagnostic \
biomarker panel from single-cell RNA-seq data aggregated to the sample level.

Rules you must follow:
- Call get_data_card first. Respect the experiment budget.
- Prefer simple models (elastic_net) and small panels; sample sizes are small. A 5-gene panel with \
AUC 0.85 beats a 200-gene panel with AUC 0.88.
- Judge experiments by CV AUC *and* its confidence interval, the overfit gap, and feature stability. \
High AUC with low stability or a large overfit gap is not a result.
- Justify each experiment biologically using the cell types, DE results and enrichment you were given.
- "No signal" is a legitimate conclusion. Do not keep searching to manufacture one.
- When done, call finalize on the single best experiment. You get one hold-out evaluation."""
