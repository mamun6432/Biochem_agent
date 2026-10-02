"""Two tests that catch most agentic-ML failure modes:

1. Shuffled labels -> the harness must report no_signal (AUC CI includes 0.5).
   If this fails, something is leaking.
2. Planted biomarker -> the harness must recover it in the right cell type,
   with stable features and a hold-out AUC well above 0.5.
"""
import numpy as np
import pandas as pd
import pytest

from agentic_ml import Harness, ExperimentSpec, build_sample_dataset
from agentic_ml.simulate import simulate_cells


def _dataset(shuffle=False, seed=0, **kw):
    counts, obs, genes = simulate_cells(seed=seed, **kw)
    if shuffle:
        rng = np.random.default_rng(seed + 1)
        samples = obs["sample"].unique()
        new_cond = dict(zip(samples, rng.permutation(obs.groupby("sample")["condition"].first().values)))
        obs["condition"] = obs["sample"].map(new_cond)  # shuffle at SAMPLE level
    return build_sample_dataset(counts, obs, genes, "sample", "condition", "cell_type",
                                positive_label="disease")


SPEC = dict(name="enet_ct0", rationale="elastic net on ct0 pseudobulk; testing recovery",
            feature_blocks=["pseudobulk"], selection="univariate_f", n_features=20,
            model="elastic_net", hyperparams={"C": 0.5, "l1_ratio": 0.5})


def test_shuffled_labels_no_signal():
    n_sig = 0
    for seed in range(3):
        h = Harness(_dataset(shuffle=True, seed=seed, n_samples=32), budget=5, seed=seed)
        r = h.run_experiment(ExperimentSpec(**SPEC))
        assert r.status in ("ok", "no_signal")
        n_sig += r.status == "ok"
        assert r.cv_auc < 0.8, f"suspiciously high AUC on shuffled labels: {r.cv_auc}"
    assert n_sig <= 1, "signal found on shuffled labels in most seeds -> leakage"


def test_recovers_planted_biomarker():
    h = Harness(_dataset(n_samples=32, effect=2.0), budget=5, seed=0)
    r = h.run_experiment(ExperimentSpec(**SPEC))
    assert r.status == "ok"
    assert r.cv_auc > 0.8
    top = [t["feature"] for t in r.top_features[:8]]
    planted = {f"ct0|G{i}" for i in range(5)}
    assert len(planted & set(top)) >= 3, f"planted markers not recovered: {top}"
    fin = h.finalize(ExperimentSpec(**SPEC))
    assert fin["holdout_auc"] > 0.7


def test_guards():
    h = Harness(_dataset(n_samples=32), budget=1, seed=0)
    assert h.run_experiment(ExperimentSpec(name="x", rationale="short")).status == "rejected"
    assert h.run_experiment(ExperimentSpec(**SPEC)).status != "rejected"
    assert "budget" in h.run_experiment(ExperimentSpec(**SPEC)).message
    h.finalize(ExperimentSpec(**SPEC))
    with pytest.raises(RuntimeError):
        h.finalize(ExperimentSpec(**SPEC))


def test_too_few_samples_rejected():
    h = Harness(_dataset(n_samples=6), holdout_frac=0.0, budget=5, seed=0)
    r = h.run_experiment(ExperimentSpec(**SPEC))
    assert r.status == "rejected" and "samples" in r.message
