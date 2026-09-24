"""Cross-encoder rerank providers shared by the reflex router and recall_v2.

ANCHOR_RERANK_PROVIDER selects the service:
  voyage       (default) Voyage rerank-2.5-lite; key from VOYAGE_API_KEY or VOYAGE_KEY_FILE
  siliconflow  SiliconFlow BAAI/bge-reranker-v2-m3, reachable from mainland China;
               key from ANCHOR_RERANK_API_KEY or SILICONFLOW_API_KEY
ANCHOR_RERANK_MODEL and ANCHOR_RERANK_URL override the provider defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULTS = {
    # provider: (url, model, top-k field in the request, results field in the response)
    "voyage": ("https://api.voyageai.com/v1/rerank", "rerank-2.5-lite", "top_k", "data"),
    "siliconflow": ("https://api.siliconflow.cn/v1/rerank", "BAAI/bge-reranker-v2-m3", "top_n", "results"),
}


def provider() -> str:
    name = os.environ.get("ANCHOR_RERANK_PROVIDER", "voyage").strip().lower()
    return name if name in _DEFAULTS else "voyage"


def model_name() -> str:
    return os.environ.get("ANCHOR_RERANK_MODEL", "").strip() or _DEFAULTS[provider()][1]


def _voyage_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    key_file = os.environ.get("VOYAGE_KEY_FILE", "").strip()
    if not key and key_file:
        secret = Path(key_file).expanduser()
        if secret.is_file():
            for line in secret.read_text(encoding="utf-8").splitlines():
                if line.startswith("VOYAGE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return key


def api_key() -> str:
    if provider() == "siliconflow":
        return (os.environ.get("ANCHOR_RERANK_API_KEY") or os.environ.get("SILICONFLOW_API_KEY") or "").strip()
    return _voyage_key()


def request(query: str, documents: list[str], top_k: int):
    """(url, headers, body) for one rerank call, or None when no key is configured."""
    key = api_key()
    if not key:
        return None
    url, _, top_field, _ = _DEFAULTS[provider()]
    url = os.environ.get("ANCHOR_RERANK_URL", "").strip() or url
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model_name(), "query": query, "documents": documents, top_field: top_k}
    return url, headers, body


def scores(payload: dict, count: int) -> list[float]:
    """Relevance scores aligned with the original document order, clamped to 0..1."""
    results_field = _DEFAULTS[provider()][3]
    score_map = {
        int(item["index"]): float(item["relevance_score"])
        for item in (payload or {}).get(results_field, [])
        if "index" in item and "relevance_score" in item
    }
    return [max(0.0, min(1.0, score_map.get(i, 0.0))) for i in range(count)]
