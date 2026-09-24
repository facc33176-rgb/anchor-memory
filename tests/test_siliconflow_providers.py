import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rerank_providers
from anchor_memory import OpenAICompatibleEmbedder


def test_rerank_defaults_to_voyage_and_needs_a_key(monkeypatch):
    for name in ("ANCHOR_RERANK_PROVIDER", "VOYAGE_API_KEY", "VOYAGE_KEY_FILE", "ANCHOR_RERANK_MODEL", "ANCHOR_RERANK_URL"):
        monkeypatch.delenv(name, raising=False)
    assert rerank_providers.provider() == "voyage"
    assert rerank_providers.model_name() == "rerank-2.5-lite"
    assert rerank_providers.request("q", ["a"], 1) is None
    monkeypatch.setenv("VOYAGE_API_KEY", "v-key")
    url, headers, body = rerank_providers.request("q", ["a", "b"], 2)
    assert url == "https://api.voyageai.com/v1/rerank"
    assert headers["Authorization"] == "Bearer v-key"
    assert body == {"model": "rerank-2.5-lite", "query": "q", "documents": ["a", "b"], "top_k": 2}
    assert rerank_providers.scores({"data": [{"index": 1, "relevance_score": 0.9}]}, 2) == [0.0, 0.9]


def test_siliconflow_rerank_request_and_scores(monkeypatch):
    monkeypatch.setenv("ANCHOR_RERANK_PROVIDER", "siliconflow")
    monkeypatch.delenv("ANCHOR_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("ANCHOR_RERANK_MODEL", raising=False)
    monkeypatch.delenv("ANCHOR_RERANK_URL", raising=False)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "s-key")
    url, headers, body = rerank_providers.request("q", ["a", "b", "c"], 3)
    assert url == "https://api.siliconflow.cn/v1/rerank"
    assert headers["Authorization"] == "Bearer s-key"
    assert body == {"model": "BAAI/bge-reranker-v2-m3", "query": "q", "documents": ["a", "b", "c"], "top_n": 3}
    payload = {"results": [{"index": 2, "relevance_score": 1.4}, {"index": 0, "relevance_score": 0.3}]}
    assert rerank_providers.scores(payload, 3) == [0.3, 0.0, 1.0]


def test_openai_compatible_embedder_shapes(monkeypatch):
    monkeypatch.setenv("ANCHOR_EMBEDDING_API_KEY", "e-key")
    monkeypatch.delenv("ANCHOR_EMBEDDING_URL", raising=False)
    monkeypatch.delenv("ANCHOR_EMBEDDING_MODEL", raising=False)
    sent = []

    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        sent.append((req.full_url, req.headers["Authorization"], body))
        data = [{"index": i, "embedding": [float(i), 1.0, 2.0]} for i in reversed(range(len(body["input"])))]
        return io.BytesIO(json.dumps({"data": data}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    emb = OpenAICompatibleEmbedder("siliconflow")
    one = emb.encode("你好")
    many = emb.encode(["a", "b"])
    assert one.shape == (3,)
    assert many.shape == (2, 3) and many[1][0] == 1.0
    assert emb.encode_query("q").shape == (3,)
    assert sent[0][0] == "https://api.siliconflow.cn/v1/embeddings"
    assert sent[0][1] == "Bearer e-key"
    assert sent[0][2]["model"] == "BAAI/bge-m3"
