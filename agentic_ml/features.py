# developer Abdullah Al Mamun

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SampleDataset:
    """Everything the harness needs, at sample resolution."""
    blocks: dict[str, pd.DataFrame]   # block name -> (samples x features)
    y: pd.Series                      # sample -> 0/1 label
    sample_meta: pd.DataFrame         # sample -> metadata (batch, etc.)
    cell_types: list[str]
    genes: list[str]

    @property
    def samples(self) -> pd.Index:
        return self.y.index

    def matrix(self, blocks: list[str], cell_types=None, genes=None) -> pd.DataFrame:
        """Assemble a feature matrix from selected blocks with optional filters."""
        parts = []
        for b in blocks:
            df = self.blocks[b]
            if b == "pseudobulk":
                cols = df.columns
                if cell_types is not None:
                    cols = [c for c in cols if c.split("|")[0] in set(cell_types)]
                if genes is not None:
                    g = set(genes)
                    cols = [c for c in cols if c.split("|")[1] in g]
                df = df[list(cols)]
            elif b == "proportions" and cell_types is not None:
                df = df[[c for c in df.columns if c.split("|")[1] in set(cell_types)]]
            parts.append(df)
        X = pd.concat(parts, axis=1).loc[self.samples]
        return X


def build_sample_dataset(
    counts,                      # cells x genes, np.ndarray or scipy sparse
    obs: pd.DataFrame,           # cell metadata
    var_names,                   # gene names
    sample_col: str,
    label_col: str,
    celltype_col: str,
    positive_label,
    min_cells_per_type: int = 10,
    geneset_scores: pd.DataFrame | None = None,   # optional cells x genesets
) -> SampleDataset:
    obs = obs.copy()
    genes = list(var_names)
    samples = sorted(obs[sample_col].unique())
    cell_types = sorted(obs[celltype_col].unique())

    # labels: one per sample; error if a sample has mixed labels
    lab = obs.groupby(sample_col)[label_col].agg(lambda s: s.unique())
    mixed = [s for s, v in lab.items() if len(v) > 1]
    if mixed:
        raise ValueError(f"samples with mixed labels: {mixed}")
    y = pd.Series({s: int(v[0] == positive_label) for s, v in lab.items()}).loc[samples]

    # --- pseudobulk: mean library-normalized expression per sample x cell type
    dense = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    lib = dense.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    cpm = dense / lib * 1e4

    pb = {}
    n_cells = {}
    for s in samples:
        for ct in cell_types:
            mask = ((obs[sample_col] == s) & (obs[celltype_col] == ct)).to_numpy()
            n = int(mask.sum())
            n_cells[(s, ct)] = n
            if n >= min_cells_per_type:
                pb[(s, ct)] = cpm[mask].mean(axis=0)
            else:
                pb[(s, ct)] = np.full(len(genes), np.nan)
    pb_df = pd.DataFrame(
        {f"{ct}|{g}": [pb[(s, ct)][i] for s in samples]
         for ct in cell_types for i, g in enumerate(genes)},
        index=samples,
    )
    # cell types missing in some samples: impute column-wise with median (harness re-scales)
    pb_df = pb_df.fillna(pb_df.median())

    # --- proportions
    ct_counts = pd.DataFrame(
        {f"prop|{ct}": [n_cells[(s, ct)] for s in samples] for ct in cell_types},
        index=samples,
    )
    prop_df = ct_counts.div(ct_counts.sum(axis=1), axis=0)

    blocks = {"pseudobulk": pb_df, "proportions": prop_df}

    if geneset_scores is not None:
        gs = geneset_scores.groupby(obs[sample_col].values).mean().loc[samples]
        gs.columns = [f"gs|{c}" for c in gs.columns]
        blocks["geneset_scores"] = gs

    sample_meta = obs.groupby(sample_col).first().loc[samples]
    sample_meta["n_cells"] = obs.groupby(sample_col).size().loc[samples]

    return SampleDataset(blocks=blocks, y=y, sample_meta=sample_meta,
                         cell_types=cell_types, genes=genes)


def from_anndata(adata, sample_col, label_col, celltype_col, positive_label,
                 layer=None, **kw) -> SampleDataset:
    X = adata.layers[layer] if layer else adata.X
    return build_sample_dataset(X, adata.obs, adata.var_names, sample_col,
                                label_col, celltype_col, positive_label, **kw)
