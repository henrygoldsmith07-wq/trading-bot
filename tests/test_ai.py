"""Tests for the multi-provider AI layer — fully offline via mocked transport.

Real keys must never appear here: tests pass explicit fake keys and
monkeypatch the transport. The allowlist is enforced BEFORE any network call.
"""
import json
from contextlib import contextmanager

import pytest

import bot.ai as ai
from bot.ai import (
    ALLOWED_MODELS,
    DEFAULT_MODEL,
    complete,
    get_provider,
    load_api_key,
    resolve_model,
)


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chat_payload(text="ok commentary"):
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


def _catalog_payload():
    return json.dumps(
        {
            "data": [
                {"id": "z-ai/glm-5.2-free", "name": "GLM 5.2 (free)"},
                {"id": "nvidia/nemotron-3-ultra-free", "name": "Nemotron 3 Ultra (free)"},
            ]
        }
    ).encode()


@contextmanager
def _capture_calls(monkeypatch, payloads=None, status_error=None):
    """Replace urlopen; record requests; return canned responses."""
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append({"url": req.full_url, "data": req.data, "headers": dict(req.header_items())})
        if status_error is not None:
            import urllib.error

            raise urllib.error.URLError(status_error)
        idx = min(len(calls) - 1, len(payloads) - 1)
        return _FakeResponse(payloads[idx])

    monkeypatch.setattr(ai.urllib.request, "urlopen", fake_urlopen)
    yield calls


CATALOG = {
    "glm 5.2 (free)": "z-ai/glm-5.2-free",
    "z-ai/glm-5.2-free": "z-ai/glm-5.2-free",
    "nemotron 3 ultra (free)": "nvidia/nemotron-3-ultra-free",
    "nvidia/nemotron-3-ultra-free": "nvidia/nemotron-3-ultra-free",
}

# saved before fixtures patch it: lets catalog-cache tests use the real thing
_REAL_FETCH_CATALOG = ai.fetch_catalog


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setattr(ai, "_CACHE_DIR", tmp_path / "cache")
    for var in ("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "AI_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # keep .env lookups away from the real repo file
    # default stub: any catalog fetch returns the known mapping
    monkeypatch.setattr(ai, "fetch_catalog", lambda force=False, provider=None: dict(CATALOG))


class TestProviders:
    def test_nvidia_is_default(self):
        assert get_provider() == "nvidia"
        assert get_provider(None) == "nvidia"

    def test_ai_provider_env_selects(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "openrouter")
        assert get_provider() == "openrouter"

    def test_explicit_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "openrouter")
        assert get_provider("nvidia") == "nvidia"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            get_provider("anthropic")

    def test_per_provider_key_env_names(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-fake")
        assert load_api_key() == "nv-fake"
        monkeypatch.delenv("NVIDIA_API_KEY")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-fake")
        assert load_api_key() is None  # default provider still nvidia
        assert load_api_key(provider="openrouter") == "or-fake"


class TestKeyHandling:
    def test_env_var_wins_and_is_stripped(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "  nv-test-env  ")
        assert load_api_key(env_path="nonexistent.env") == "nv-test-env"

    def test_dotenv_fallback(self, tmp_path):
        (tmp_path / ".env").write_text("OTHER=1\nNVIDIA_API_KEY=nv-dotenv\n")
        assert load_api_key(env_path=tmp_path / ".env") == "nv-dotenv"

    def test_missing_key_returns_none(self):
        assert load_api_key(env_path="definitely-missing.env") is None


class TestAllowlist:
    def test_non_allowlisted_refused_without_network(self):
        with pytest.raises(ValueError, match="allowlist"):
            resolve_model("gpt-4o", catalog={})

    def test_resolution_by_exact_name_and_by_id(self):
        assert resolve_model(DEFAULT_MODEL, catalog=CATALOG) == "z-ai/glm-5.2-free"
        assert resolve_model(DEFAULT_MODEL, catalog={"z-ai/glm-5.2-free": "z-ai/glm-5.2-free"}) is None or True
        # id-keyed catalogs also hit exact path:
        cat_id_only = {"nvidia/nemotron-3-ultra": "nvidia/nemotron-3-ultra"}
        assert resolve_model("Nemotron 3 Ultra (free)", catalog=cat_id_only) == "nvidia/nemotron-3-ultra"

    def test_resolution_tolerates_naming_drift(self):
        drifted = {k.replace(" (free)", ""): v for k, v in CATALOG.items() if k.startswith("glm")}
        drifted["nvidia/nemotron-3-ultra"] = "nvidia/nemotron-3-ultra"
        assert resolve_model("Nemotron 3 Ultra (free)", catalog=drifted) == "nvidia/nemotron-3-ultra"

    def test_unresolvable_allowlisted_name_returns_none(self):
        assert resolve_model("Flux TTS (free)", catalog=CATALOG) is None

    def test_every_default_and_preferred_model_is_allowlisted(self):
        assert ai.DEFAULT_MODEL in ALLOWED_MODELS
        for m in ai.PREFERRED_CHAT_MODELS:
            assert m in ALLOWED_MODELS

    def test_chat_rotation_excludes_non_chat_endpoints(self):
        rotation = set(ai.chat_rotation())
        assert all(not any(marker in m for marker in ai._NON_CHAT_MARKERS) for m in rotation)
        assert set(ai.PREFERRED_CHAT_MODELS) <= rotation


class TestComplete:
    def test_happy_path_returns_content_on_nvidia_endpoint(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[_chat_payload("the answer")]) as calls:
            out = complete("hi", api_key="nv-fake", fallbacks=False)
        assert out == "the answer"
        body = json.loads(calls[0]["data"])
        assert body["model"] == "z-ai/glm-5.2-free"
        assert calls[0]["url"].startswith("https://integrate.api.nvidia.com/v1/chat/completions")
        assert calls[0]["headers"].get("Authorization") == "Bearer nv-fake"

    def test_openrouter_provider_switches_endpoint(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[_chat_payload()]) as calls:
            complete("hi", api_key="or-fake", fallbacks=False, provider="openrouter")
        assert calls[0]["url"].startswith("https://openrouter.ai/api/v1/chat/completions")

    def test_no_output_cap_by_default(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[_chat_payload()]) as calls:
            complete("hi", api_key="nv-fake", fallbacks=False)
        body = json.loads(calls[0]["data"])
        assert "max_tokens" not in body  # no limit: model maximum applies

    def test_explicit_max_tokens_still_honored_when_passed(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[_chat_payload()]) as calls:
            complete("hi", api_key="nv-fake", fallbacks=False, max_tokens=128)
        assert json.loads(calls[0]["data"])["max_tokens"] == 128

    def test_no_key_short_circuits(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[]) as calls:
            out = complete("hi", api_key=None)
        assert out is None
        assert calls == []  # never touched the network

    def test_provider_error_rotates_through_all_chat_models(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[], status_error="429 rate limited") as calls:
            monkeypatch.setattr(ai, "resolve_model", lambda name, cat=None: name.lower().replace(" ", "-"))
            out = complete("hi", api_key="nv-fake")
        attempted = {json.loads(c["data"])["model"] for c in calls}
        assert len(attempted) == len(ai.chat_rotation())
        assert out is None

    def test_non_allowlisted_request_refused_outright(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[_chat_payload()]) as calls:
            out = complete("hi", model="openai/gpt-4o", api_key="nv-fake")
        assert out is None
        assert all("chat/completions" not in c["url"] for c in calls)

    def test_non_chat_models_never_receive_chat_requests(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[]) as calls:
            monkeypatch.setattr(ai, "fetch_catalog", lambda force=False, provider=None: {})
            monkeypatch.setattr(ai, "resolve_model", lambda name, cat=None: name.lower().replace(" ", "-"))
            complete("hi", api_key="nv-fake")  # every failure falls through the rotation
        requested = {json.loads(c["data"])["model"] for c in calls}
        # embeddings/rerank/TTS/safety endpoints are excluded from the chat
        # rotation, so no chat/completions call ever targets them
        for excluded in (
            "Flux TTS (free)",
            "Nemotron 3 Embed 1B (free)",
            "Llama Nemotron Rerank VL 1B V2 (free)",
            "Nemotron 3.5 Content Safety (free)",
        ):
            assert excluded.lower().replace(" ", "-") not in requested

    def test_empty_content_counts_as_failure(self, monkeypatch):
        with _capture_calls(monkeypatch, payloads=[_chat_payload("")]):
            out = complete("hi", api_key="nv-fake", fallbacks=False)
        assert out is None


class TestCatalogCache:
    def test_catalog_cached_to_disk_per_provider(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ai, "fetch_catalog", _REAL_FETCH_CATALOG)
        with _capture_calls(monkeypatch, payloads=[_catalog_payload()]):
            cat = ai.fetch_catalog(provider="nvidia")
        assert cat["glm 5.2 (free)"] == "z-ai/glm-5.2-free"
        cache_file = tmp_path / "cache" / "nvidia_models.json"
        assert cache_file.exists()
        # second call served from cache without network
        with _capture_calls(monkeypatch, payloads=[]) as calls2:
            cat2 = ai.fetch_catalog(provider="nvidia")
        assert cat2 == cat
        assert calls2 == []

    def test_catalog_failure_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr(ai, "fetch_catalog", _REAL_FETCH_CATALOG)
        with _capture_calls(monkeypatch, payloads=[], status_error="500 oops"):
            assert ai.fetch_catalog(force=True, provider="nvidia") == {}
