"""Shared Streamlit widget: pick an LLM provider/model and take the viewer's own API key."""
from __future__ import annotations

import os

import streamlit as st

from agentic_ml.providers import PRESETS, ollama_models

ALLOW_ENV_KEYS = os.environ.get("AGENTIC_ML_ALLOW_ENV_KEYS") == "1"


@st.cache_data(ttl=15, show_spinner=False)
def _ollama_models(url: str) -> list[str]:
    return ollama_models(url)


def agent_settings() -> dict:
    """Sidebar block: provider, endpoint, model and the viewer's own API key."""
    st.subheader("Model")
    pid = st.selectbox("Provider", list(PRESETS), format_func=lambda k: PRESETS[k].label)
    p = PRESETS[pid]
    base_url = p.base_url
    if pid in ("ollama", "custom"):
        base_url = st.text_input("Server URL", p.base_url, key=f"url_{pid}")

    if pid == "ollama":
        local = _ollama_models(base_url)
        if local:
            default = local.index(p.models[0]) if p.models[0] in local else 0
            model = st.selectbox("Model", local, index=default, key="model_ollama",
                                 help=f"Pulled models on {base_url}. Get more with `ollama pull <name>`.")
        else:
            st.warning(f"Can't reach Ollama at {base_url}. Start it (`ollama serve`) and pull a model "
                       f"(`ollama pull {p.models[0]}`).")
            model = st.text_input("Model", p.models[0], key="model_ollama_txt")
    elif p.models:
        model = st.selectbox("Model", p.models, accept_new_options=True, key=f"model_{pid}",
                             help="Pick one or type any model ID.")
    else:
        model = st.text_input("Model ID", key=f"model_{pid}", placeholder="as listed in the provider's docs")

    key = ""
    if p.needs_key or pid == "custom":
        key = st.text_input(f"{p.label} API key" + ("" if p.needs_key else " (optional)"),
                            type="password", key=f"key_{pid}")
        if not key and ALLOW_ENV_KEYS and p.key_env and os.environ.get(p.key_env):
            st.caption(f"Blank key → using ${p.key_env} from this machine.")
        st.caption("🔒 Your key is kept only in this browser session's memory. "
                   "It is never written to disk or the ledger.")
    return {"provider": pid, "model": (model or "").strip(), "base_url": base_url, "api_key": key}
