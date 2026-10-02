"""Synthetic scRNA-seq-like data with a planted biomarker, for testing the harness."""
import numpy as np
import pandas as pd


def simulate_cells(n_samples=24, cells_per_sample=150, n_genes=300, n_cell_types=4, 
                   n_marker_genes=5, effect=1.5, disease_celltype="ct0", seed=0):
    """Returns counts (cells x genes), obs, var_names. First n_marker_genes are
    up-regulated in `disease_celltype` of diseased samples only."""
    rng = np.random.default_rng(seed)
    cell_types = [f"ct{i}" for i in range(n_cell_types)]
    genes = [f"G{i}" for i in range(n_genes)]
    base = rng.gamma(2.0, 1.0, size=n_genes)                    # gene means
    ct_shift = rng.normal(0, 0.5, size=(n_cell_types, n_genes))  # cell-type identity
    rows, obs = [], []
    for s in range(n_samples):
        disease = s % 2
        sample_eff = rng.normal(0, 0.3, size=n_genes)            # patient noise
        for _ in range(cells_per_sample):
            ct = rng.integers(n_cell_types)
            mu = base * np.exp(ct_shift[ct] + sample_eff)
            if disease and cell_types[ct] == disease_celltype:
                mu[:n_marker_genes] *= effect
            rows.append(rng.poisson(mu))
            obs.append({"sample": f"S{s:02d}", "condition": "disease" if disease else "healthy",
                        "cell_type": cell_types[ct]})
    return np.vstack(rows), pd.DataFrame(obs), genes
