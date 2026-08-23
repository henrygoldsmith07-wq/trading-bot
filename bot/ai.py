"""Optional AI layer via OpenAI-compatible providers — advisory commentary only.

Providers:
- "nvidia"      (default) https://integrate.api.nvidia.com/v1  — NVIDIA_API_KEY
- "openrouter"            https://openrouter.ai/api/v1         — OPENROUTER_API_KEY

Select with the AI_PROVIDER environment variable (or a line in .env).
Ground rules enforced by design, for every provider:

- API keys live in the environment or a gitignored `.env` file. They are
  never hardcoded and never logged.
- Only allowlisted free models may be called. The allowlist is stored as
  display names (as shown on OpenRouter / NVIDIA build) and resolved against
  the live model catalog at runtime (disk-cached per provider); anything
  outside the list is refused without a network call.
- Chat requests rotate through EVERY allowlisted chat-capable model when one
  fails or is rate-limited; embeddings/rerank/TTS/safety endpoints are
  excluded from the chat rotation automatically.
- AI output is commentary on top of deterministic numbers. No code path in
  this repo feeds model output back into weights, signals, or decisions.
- Every entry point degrades gracefully: no key / no catalog / provider
  error all return None and callers proceed without commentary.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROVIDERS: dict[str, dict[str, str]] = {
    "nvidia": {"base": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY"},
    "openrouter": {"base": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
}
DEFAULT_PROVIDER = "nvidia"
_CATALOG_TTL_SECONDS = 24 * 3600
_CACHE_DIR = Path(".cache")

# Display names exactly as listed on OpenRouter / NVIDIA build (user-approved).
ALLOWED_MODELS: tuple[str, ...] = (
    "Ox Alpha",
    "LFM2.5-Embedding-350M (free)",
    "Dots3-Note Preview (free)",
    "Flux TTS (free)",
    "LFM2.5-2.6B (free)",
    "Nemotron 3.5 Lightning (free)",
    "Inkling Small (free)",
    "S2.1 Pro Free (free)",
    "Laguna S 2.1 (free)",
    "Inkling (free)",
    "Nemotron 3 Embed 1B (free)",
    "Laguna XS 2.1 (free)",
    "North Mini Code (free)",
    "GLM 5.2 (free)",
    "Llama Nemotron Rerank VL 1B V2 (free)",
    "Nemotron 3.5 Content Safety (free)",
    "Nemotron 3 Ultra (free)",
    "Nemotron 3 Nano Omni (free)",
    "Gemma 4 26B A4B (free)",
    "Gemma 4 31B (free)",
    "Nemotron 3 Super (free)",
    "Llama Nemotron Embed VL 1B V2 (free)",
    "Nemotron 3 Nano 30B A3B (free)",
    "Nemotron Nano 12B 2 VL (free)",
    "Nemotron Nano 9B V2 (free)",
)

DEFAULT_MODEL = "GLM 5.2 (free)"

# Models tried first for chat, in order (strongest general text models on the
# allowlist). After these are exhausted, rotation continues through every
# remaining allowlisted model that speaks chat/completions — see _NON_CHAT.
PREFERRED_CHAT_MODELS: tuple[str, ...] = (
    "GLM 5.2 (free)",
    "Nemotron 3 Ultra (free)",
    "Gemma 4 31B (free)",
    "Nemotron 3 Super (free)",
    "Nemotron 3.5 Lightning (free)",
    "Gemma 4 26B A4B (free)",
    "Nemotron Nano 9B V2 (free)",
    "LFM2.5-2.6B (free)",
)

# Endpoint families that cannot serve /chat/completions; excluded from the
# chat rotation but still allowlisted for dedicated integrations later.
_NON_CHAT_MARKERS = ("Embed", "Rerank", "TTS", "Content Safety")


def get_provider(provider: str | None = None) -> str:
    """Provider name from the argument, AI_PROVIDER setting, or default."""
    name = (provider or os.environ.get("AI_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"unknown AI provider {name!r}; choose from {sorted(PROVIDERS)}")
    return name


def load_api_key(env_path: str | Path = ".env", provider: str | None = None) -> str | None:
    """API key from the environment first, then a gitignored .env file."""
    prov = PROVIDERS[get_provider(provider)]
    key = os.environ.get(prov["key_env"])
    if key:
        return key.strip()
    p = Path(env_path)
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{prov['key_env']}="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def _catalog_cache_path(provider: str) -> Path:
    return _CACHE_DIR / f"{provider}_models.json"


def fetch_catalog(force: bool = False, provider: str | None = None) -> dict[str, str]:
    """Model catalog as {lookup_name_lower: model_id}, cached per provider.

    Both the display name and the id itself index to the id, so resolvers can
    match either form. Returns an empty dict on any failure so callers
    degrade gracefully.
    """
    prov = get_provider(provider)
    cache = _catalog_cache_path(prov)
    if not force and cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if time.time() - blob.get("fetched_at", 0) < _CATALOG_TTL_SECONDS:
                return blob["models"]
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            pass
    headers = {"User-Agent": "trading-bot-research"}
    key = load_api_key(provider=prov)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        req = urllib.request.Request(f"{PROVIDERS[prov]['base']}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode())
        models: dict[str, str] = {}
        for m in payload.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            models[str(m.get("name") or mid).strip().lower()] = mid
            models[str(mid).strip().lower()] = mid
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"fetched_at": time.time(), "models": models}))
        return models
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return {}


def resolve_model(display_name: str, catalog: dict[str, str] | None = None) -> str | None:
    """Map an allowlisted display name to a provider model id.

    Matching order: exact lowercased name/id, then distinctive-token match
    with parenthesized qualifiers ((free), (preview)) treated as optional.
    Ambiguity resolves deterministically (first sorted id).
    """
    if display_name not in ALLOWED_MODELS:
        raise ValueError(f"model {display_name!r} is not on the approved allowlist")
    cat = catalog if catalog is not None else fetch_catalog()
    hit = cat.get(display_name.lower())
    if hit:
        return hit
    core = display_name.lower().replace("(", " ").replace(")", " ").split()
    qualifiers = {"free", "preview"}
    strict_tokens = [t for t in core if len(t) > 2]
    loose_tokens = [t for t in strict_tokens if t not in qualifiers]
    if not loose_tokens:
        return None
    for need_all in (strict_tokens, loose_tokens):
        matches = sorted({mid for name, mid in cat.items() if all(t in name for t in need_all)})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[0]
    return None


def chat_rotation() -> tuple[str, ...]:
    """All allowlisted models eligible for chat/completions: preferred order
    first, then every other text model — no hard cap on how many may serve."""
    preferred = [m for m in PREFERRED_CHAT_MODELS if m in ALLOWED_MODELS]
    rest = [
        m
        for m in ALLOWED_MODELS
        if m not in preferred and not any(marker in m for marker in _NON_CHAT_MARKERS)
    ]
    return tuple(preferred + rest)


def complete(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
    timeout: int = 60,
    fallbacks: bool = True,
    provider: str | None = None,
) -> str | None:
    """One chat completion using an allowlisted model on the chosen provider.

    `max_tokens=None` (the default) imposes no output cap — the model's own
    maximum applies. When the requested model fails or is rate-limited, the
    request rotates through every other allowlisted chat model (`fallbacks`),
    so per-model free-tier limits do not stop the layer.

    Returns the assistant text, or None when unavailable (no key, unknown
    models, provider errors) — callers must treat output as optional garnish.
    """
    prov = get_provider(provider)
    base = PROVIDERS[prov]["base"]
    key = api_key or load_api_key(provider=prov)
    if not key:
        return None
    requested = model or DEFAULT_MODEL
    if requested not in ALLOWED_MODELS:
        print(f"[ai] {requested!r} is not on the approved allowlist; refusing")
        return None
    rotation = [requested]
    if fallbacks:
        rotation += [m for m in chat_rotation() if m != requested]
    catalog = fetch_catalog(provider=prov)
    last_error = ""
    for candidate in rotation:
        model_id = resolve_model(candidate, catalog)
        if model_id is None:
            last_error = f"unresolved: {candidate}"
            continue
        body: dict = {
            "model": model_id,
            "temperature": temperature,
            "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/henrygoldsmith07-wq/trading-bot",
                "X-Title": "trading-bot research",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, str) and content.strip():
                return content.strip()
            last_error = f"empty response from {model_id}"
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError, IndexError) as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
    print(f"[ai] unavailable ({last_error}); continuing without commentary")
    return None
