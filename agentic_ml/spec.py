"""Experiment specification: the only thing the agent is allowed to write.

Everything the agent controls lives here. Everything it must NOT control
(split logic, held-out evaluation, budgets) lives in harness.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json

ModelName = Literal["elastic_net", "random_forest", "gradient_boosting", "linear_svm"]
FeatureBlock = Literal["pseudobulk", "proportions", "geneset_scores"]
SelectionMethod = Literal["none", "univariate_f", "l1"]


@dataclass
class ExperimentSpec:
    name: str
    rationale: str  # agent must justify the experiment (biology + prior results)

    # --- feature representation ---------------------------------------
    feature_blocks: list[FeatureBlock] = field(default_factory=lambda: ["pseudobulk"])
    cell_types: Optional[list[str]] = None      # None = all annotated cell types
    genes: Optional[list[str]] = None           # None = all genes in the feature matrix
    log_transform: bool = True
    scale: bool = True

    # --- feature selection (always applied INSIDE each CV fold) --------
    selection: SelectionMethod = "none"
    n_features: Optional[int] = None            # for univariate_f

    # --- model -----------------------------------------------------------
    model: ModelName = "elastic_net"
    hyperparams: dict = field(default_factory=dict)  # fixed values
    tune: dict = field(default_factory=dict)         # name -> list of candidates (inner CV)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentSpec":
        # null means "use the default", except for fields where None is itself meaningful
        nullable = {"cell_types", "genes", "n_features"}
        return cls(**{k: v for k, v in d.items() if v is not None or k in nullable})

    def validate(self) -> list[str]:
        problems = []
        if not self.rationale or len(self.rationale) < 20:
            problems.append("rationale must be a real justification (>=20 chars)")
        if self.selection == "univariate_f" and not self.n_features:
            problems.append("n_features required for univariate_f selection")
        if not self.feature_blocks:
            problems.append("at least one feature block required")
        return problems


# JSON schema handed to the LLM so it produces valid specs.
SPEC_JSON_SCHEMA = {
    "type": "object",
    "required": ["name", "rationale"],
    "properties": {
        "name": {"type": "string"},
        "rationale": {"type": "string", "minLength": 20},
        "feature_blocks": {
            "type": "array",
            "items": {"enum": ["pseudobulk", "proportions", "geneset_scores"]},
        },
        "cell_types": {"type": ["array", "null"], "items": {"type": "string"}},
        "genes": {"type": ["array", "null"], "items": {"type": "string"}},
        "log_transform": {"type": "boolean"},
        "scale": {"type": "boolean"},
        "selection": {"enum": ["none", "univariate_f", "l1"]},
        "n_features": {"type": ["integer", "null"], "minimum": 1},
        "model": {"enum": ["elastic_net", "random_forest", "gradient_boosting", "linear_svm"]},
        "hyperparams": {
            "type": "object",
            "description": "Fixed scikit-learn constructor arguments for the chosen model. "
                           "elastic_net (LogisticRegression, elasticnet): C, l1_ratio. "
                           "random_forest: n_estimators, max_depth, min_samples_leaf. "
                           "gradient_boosting: n_estimators, max_depth, learning_rate. "
                           "linear_svm: C.",
        },
        "tune": {
            "type": "object",
            "description": "Hyperparameter name -> list of candidate values, chosen by inner CV. "
                           "Same names as hyperparams, e.g. {\"C\": [0.1, 1, 10]}.",
        },
    },
}
