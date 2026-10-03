# agentic_ml — bounded model search for scRNA-seq biomarker discovery

An LLM agent proposes experiments; a fixed harness scores them. The agent
controls features, models and hyperparameters. It never controls the split
logic, the sealed hold-out, or the budget.

```
agentic_ml/
  spec.py       ExperimentSpec — the only thing the agent writes (+ JSON schema)
  features.py   cells -> one row per sample (pseudobulk per cell type, proportions)
  harness.py    grouped stratified CV, in-fold selection, nested tuning, sealed hold-out, budget
  tools.py      the 4 agent tools, ledger, tool definitions, system prompt
  agent.py      provider-agnostic tool-use loop (yields events)
  providers.py  Ollama / Anthropic / OpenAI-compatible chat adapters
  preprocess.py scanpy QC -> doublets -> clustering -> marker annotation pipeline
  simulate.py   synthetic data with a planted biomarker
tests/          shuffled-label leakage test, planted-marker recovery test, guard tests
run_agent.py    CLI around the agent loop
app.py          Streamlit dashboard: preprocessing, live agent run view, ledger browser
preprocess_page.py  the app's Preprocess page
```

## Quick start
```
pip install -r requirements.txt
python -m pytest tests -q                                # harness sanity checks
python run_agent.py --provider ollama --model gpt-oss:20b   # local model, no key
ANTHROPIC_API_KEY=... python run_agent.py                # Claude on simulated data
streamlit run app.py                                     # live dashboard / ledger browser
```

## Choosing a model
`agentic_ml/providers.py` puts every LLM behind one interface. In the app pick a provider
in the sidebar; on the CLI use `--provider/--model/--base-url/--api-key`.

| Provider | Key | Notes |
|---|---|---|
| `ollama` | none | local; `ollama pull gpt-oss:20b` first. Models are listed from the running server |
| `anthropic` | user's Anthropic key | native Messages API |
| `openai`, `gemini`, `openrouter` | user's key for that service | OpenAI-compatible chat API; type any model ID |
| `custom` | optional | any OpenAI-compatible server (vLLM, LM Studio, ...) via its URL |

In the app each viewer enters their own key. It is held only in their browser session's
memory and never written to disk or the ledger. A blank key never falls back to the
server's environment unless you start the app with `AGENTIC_ML_ALLOW_ENV_KEYS=1`
(fine on your own laptop; don't set it on a shared server).

## Preprocessing (scanpy clustering tutorial, automated)
`agentic_ml/preprocess.py` runs the steps of the
[scanpy clustering tutorial](https://scanpy.scverse.org/en/stable/tutorials/basics/clustering.html):
QC metrics -> filter (min 100 genes/cell, 3 cells/gene) -> Scrublet doublets -> normalize + log1p
-> 2,000 HVGs -> PCA/neighbors/UMAP -> Leiden (0.02, 0.5, 2.0) -> marker-based cell-type labels.
Differences: predicted doublets are removed, an optional max-%-mito filter, and annotation is
automatic (each cluster gets the marker set with the highest per-gene-scaled mean expression,
i.e. what you'd read off the tutorial's dotplot). Review the `uncertain` and `mean_pct_mt`
columns before trusting the labels.
```
streamlit run app.py        # "Preprocess" mode: run, review/edit labels, save .h5ad
python -m agentic_ml.preprocess dataset/22716739/*.h5 --out dataset/processed.h5ad
```

## Using real data
1. Run upstream pipeline (QC, integration, annotation, pseudobulk DE).
2. Build a `SampleDataset` with `from_anndata(adata, sample_col, label_col, celltype_col, positive_label)`.
3. Put DE tables / enrichment results in a text file and pass `--context` so the agent can reason biologically.
4. Read `runs/ledger.jsonl` for the full search path (reviewers will ask).




