"""Evaluation harness. The agent calls run_experiment(spec); it never sees
split logic, the sealed hold-out, or the budget counter.

Developer : Abdullah Al Mamun
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

from .features import SampleDataset
from .spec import ExperimentSpec

MIN_SAMPLES_PER_CLASS = 4
TOP_K_STABILITY = 20


@dataclass
class ExperimentResult:
    name: str
    status: str                         # "ok" | "rejected" | "no_signal"
    message: str = ""
    n_samples: int = 0
    n_features: int = 0
    cv_auc: float | None = None
    cv_auc_ci: tuple[float, float] | None = None
    fold_aucs: list[float] = field(default_factory=list)
    train_auc: float | None = None
    overfit_gap: float | None = None
    brier: float | None = None
    feature_stability: float | None = None     # mean pairwise Jaccard of top-k per fold
    top_features: list[dict] = field(default_factory=list)  # [{feature, importance, folds_selected}]
    chosen_hyperparams: dict = field(default_factory=dict)
    experiments_remaining: int | None = None

    def to_dict(self):
        d = asdict(self)
        return d


def _model(spec: ExperimentSpec, seed: int):
    hp = dict(spec.hyperparams)
    if spec.model == "elastic_net":
        return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
                                  l1_ratio=hp.pop("l1_ratio", 0.5), C=hp.pop("C", 1.0),
                                  random_state=seed, **hp)
    if spec.model == "random_forest":
        return RandomForestClassifier(n_estimators=hp.pop("n_estimators", 300),
                                      max_depth=hp.pop("max_depth", None),
                                      random_state=seed, **hp)
    if spec.model == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=hp.pop("n_estimators", 100),
                                          max_depth=hp.pop("max_depth", 2),
                                          learning_rate=hp.pop("learning_rate", 0.1),
                                          random_state=seed, **hp)
    if spec.model == "linear_svm":
        return SVC(kernel="linear", probability=True, C=hp.pop("C", 1.0),
                   random_state=seed, **hp)
    raise ValueError(spec.model)


def _pipeline(spec: ExperimentSpec, seed: int) -> Pipeline:
    steps = []
    if spec.log_transform:
        steps.append(("log", FunctionTransformer(np.log1p, validate=True)))
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    if spec.selection == "univariate_f":
        steps.append(("select", SelectKBest(f_classif, k=spec.n_features)))
    elif spec.selection == "l1":
        steps.append(("select", SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=seed))))
    steps.append(("clf", _model(spec, seed)))
    return Pipeline(steps)


def _check_model_args(spec: ExperimentSpec, seed: int) -> str:
    """Catch misnamed hyperparameters up front, so a typo doesn't burn budget."""
    try:
        valid = set(_model(replace(spec, hyperparams={}), seed).get_params())
    except ValueError as e:
        return f"invalid model: {e}"
    bad = sorted((set(spec.hyperparams) | set(spec.tune)) - valid)
    if bad:
        return (f"unknown hyperparameter(s) for {spec.model}: {bad}; "
                f"valid names: {', '.join(sorted(valid))}")
    return ""


def _selected_features(pipe: Pipeline, feature_names) -> tuple[list[str], np.ndarray]:
    """Names + importances of features that survived selection in a fitted pipe."""
    names = np.asarray(feature_names)
    if "select" in pipe.named_steps:
        names = names[pipe.named_steps["select"].get_support()]
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).ravel()
    elif hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = np.zeros(len(names))
    return list(names), imp


def _bootstrap_auc_ci(y, p, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def _jaccard_stability(sets: list[set]) -> float:
    if len(sets) < 2:
        return 1.0
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
    return float(np.mean(vals))


class Harness:
    """Owns the data, the sealed hold-out, the budget, and the split logic."""

    def __init__(self, dataset: SampleDataset, holdout_frac: float = 0.25,
                 group_col: str | None = None, budget: int = 25,
                 n_splits: int = 5, seed: int = 0):
        self.seed = seed
        self.budget = budget
        self.used = 0
        self.group_col = group_col
        self._finalized = False

        rng = np.random.default_rng(seed)
        y = dataset.y
        groups = (dataset.sample_meta[group_col] if group_col else pd.Series(y.index, index=y.index))

        # sealed hold-out: stratified by label, respecting groups
        hold = set()
        for cls in np.unique(y):
            g = np.asarray(groups[y == cls].unique(), dtype=object)
            rng.shuffle(g)
            hold |= set(g[: max(1, int(round(len(g) * holdout_frac)))])
        is_hold = groups.isin(hold)
        self._dev_idx = y.index[~is_hold]
        self._hold_idx = y.index[is_hold]
        self.dataset = dataset
        self.n_splits = min(n_splits, int(y.loc[self._dev_idx].value_counts().min()))

    # ------------------------------------------------------------------ info
    def data_card(self) -> dict:
        y = self.dataset.y.loc[self._dev_idx]
        return {
            "n_dev_samples": int(len(y)),
            "n_holdout_samples": int(len(self._hold_idx)),   # count only; never the data
            "class_balance_dev": y.value_counts().to_dict(),
            "cell_types": self.dataset.cell_types,
            "n_genes": len(self.dataset.genes),
            "feature_blocks": {k: v.shape[1] for k, v in self.dataset.blocks.items()},
            "cv_folds": self.n_splits,
            "experiments_remaining": self.budget - self.used,
            "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        }

    def _xy(self, spec: ExperimentSpec, idx):
        X = self.dataset.matrix(spec.feature_blocks, spec.cell_types, spec.genes).loc[idx]
        X = X.loc[:, X.var(axis=0) > 0]  # drop constant features
        y = self.dataset.y.loc[idx].to_numpy()
        g = (self.dataset.sample_meta.loc[idx, self.group_col].to_numpy()
             if self.group_col else np.asarray(idx))
        return X, y, g

    # ------------------------------------------------------------- experiment
    def run_experiment(self, spec: ExperimentSpec) -> ExperimentResult:
        problems = spec.validate()
        if problems:
            return ExperimentResult(spec.name, "rejected", "; ".join(problems))
        if self.used >= self.budget:
            return ExperimentResult(spec.name, "rejected", "experiment budget exhausted; call finalize()")
        bad_args = _check_model_args(spec, self.seed)
        if bad_args:
            return ExperimentResult(spec.name, "rejected", bad_args + " (not charged to the budget)")
        self.used += 1

        X, y, groups = self._xy(spec, self._dev_idx)
        if pd.Series(y).value_counts().min() < MIN_SAMPLES_PER_CLASS:
            return ExperimentResult(spec.name, "rejected",
                                    f"fewer than {MIN_SAMPLES_PER_CLASS} samples in a class; "
                                    "ML is not appropriate for this dataset")
        if X.shape[1] == 0:
            return ExperimentResult(spec.name, "rejected", "feature selection left no features")

        outer = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(y))
        fold_aucs, train_aucs, top_sets, chosen = [], [], [], []
        imp_acc: dict[str, list[float]] = {}

        for tr, te in outer.split(X, y, groups):
            pipe = _pipeline(spec, self.seed)
            if spec.tune:
                grid = {f"clf__{k}": v for k, v in spec.tune.items()}
                inner_k = max(2, min(3, int(pd.Series(y[tr]).value_counts().min())))
                inner = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=self.seed)
                est = GridSearchCV(pipe, grid, cv=inner.split(X.iloc[tr], y[tr], groups[tr]),
                                   scoring="roc_auc")
                est.fit(X.iloc[tr], y[tr])
                pipe = est.best_estimator_
                chosen.append({k.replace("clf__", ""): v for k, v in est.best_params_.items()})
            else:
                pipe.fit(X.iloc[tr], y[tr])

            oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
            if len(np.unique(y[te])) == 2:
                fold_aucs.append(float(roc_auc_score(y[te], oof[te])))
            train_aucs.append(float(roc_auc_score(y[tr], pipe.predict_proba(X.iloc[tr])[:, 1])))

            names, imp = _selected_features(pipe, X.columns)
            order = np.argsort(imp)[::-1][:TOP_K_STABILITY]
            top_sets.append({names[i] for i in order if imp[i] > 0})
            for i in order:
                imp_acc.setdefault(names[i], []).append(float(imp[i]))

        cv_auc = float(roc_auc_score(y, oof))
        ci = _bootstrap_auc_ci(y, oof, seed=self.seed)
        train_auc = float(np.mean(train_aucs))
        n_folds = len(top_sets)
        top = sorted(
            ({"feature": k, "importance": float(np.mean(v)), "folds_selected": len(v) / n_folds}
             for k, v in imp_acc.items()),
            key=lambda d: (-d["folds_selected"], -d["importance"]),
        )[:TOP_K_STABILITY]

        status = "no_signal" if ci[0] <= 0.5 else "ok"
        msg = ("CV AUC confidence interval includes 0.5: no reliable signal for this configuration"
               if status == "no_signal" else "")
        return ExperimentResult(
            name=spec.name, status=status, message=msg,
            n_samples=len(y), n_features=int(X.shape[1]),
            cv_auc=cv_auc, cv_auc_ci=ci, fold_aucs=fold_aucs,
            train_auc=train_auc, overfit_gap=train_auc - cv_auc,
            brier=float(brier_score_loss(y, oof)),
            feature_stability=_jaccard_stability(top_sets),
            top_features=top,
            chosen_hyperparams=(pd.DataFrame(chosen).mode().iloc[0].to_dict() if chosen else {}),
            experiments_remaining=self.budget - self.used,
        )

    # --------------------------------------------------------------- finalize
    def finalize(self, spec: ExperimentSpec) -> dict:
        """Fit on all dev samples, evaluate ONCE on the sealed hold-out."""
        if self._finalized:
            raise RuntimeError("finalize() may only be called once")
        bad_args = _check_model_args(spec, self.seed)
        if bad_args:
            raise ValueError(bad_args)
        Xd, yd, _ = self._xy(spec, self._dev_idx)
        Xh = self.dataset.matrix(spec.feature_blocks, spec.cell_types, spec.genes).loc[self._hold_idx, Xd.columns]
        yh = self.dataset.y.loc[self._hold_idx].to_numpy()
        pipe = _pipeline(spec, self.seed)
        if spec.tune:
            grid = {f"clf__{k}": v for k, v in spec.tune.items()}
            pipe = GridSearchCV(pipe, grid, cv=self.n_splits, scoring="roc_auc").fit(Xd, yd).best_estimator_
        else:
            pipe.fit(Xd, yd)
        # the hold-out is touched only from here on; a failure above can be retried
        self._finalized = True
        p = pipe.predict_proba(Xh)[:, 1]
        names, imp = _selected_features(pipe, Xd.columns)
        order = np.argsort(imp)[::-1]
        return {
            "holdout_auc": float(roc_auc_score(yh, p)) if len(np.unique(yh)) == 2 else None,
            "holdout_auc_ci": _bootstrap_auc_ci(yh, p, seed=self.seed) if len(np.unique(yh)) == 2 else None,
            "n_holdout": int(len(yh)),
            "panel": [{"feature": names[i], "importance": float(imp[i])} for i in order if imp[i] > 0],
            "experiments_used": self.used,
        }
