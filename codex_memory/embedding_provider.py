import json
import os
import re
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass

from codex_memory.codex_data import default_codex_home


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOML_SECTION_RE = re.compile(r"^\[([^\]]+)\]\s*$")
TOML_STRING_RE = re.compile(r'^([A-Za-z0-9_.-]+)\s*=\s*"((?:[^"\\]|\\.)*)"')


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingSettings:
    api_key: str
    base_url: str
    model: str
    dimensions: int = None
    provider_name: str = "fastembed"
    effective_base_url: str = None
    local_model: str = DEFAULT_FASTEMBED_MODEL
    local_cache_dir: str = None
    effective_provider: str = None
    effective_model: str = None


def _parse_toml_string_map(path):
    if not os.path.exists(path):
        return {}
    current = ""
    values = {current: {}}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            section_match = TOML_SECTION_RE.match(line)
            if section_match:
                current = section_match.group(1)
                values.setdefault(current, {})
                continue
            kv_match = TOML_STRING_RE.match(line)
            if kv_match:
                key = kv_match.group(1)
                value = bytes(kv_match.group(2), "utf-8").decode("unicode_escape")
                values.setdefault(current, {})[key] = value
    return values


def _read_auth_key(codex_home):
    auth_path = os.path.join(codex_home, "auth.json")
    if not os.path.exists(auth_path):
        return None
    try:
        with open(auth_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    api_key = payload.get("OPENAI_API_KEY")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return None


def _read_codex_base_url(codex_home):
    config_path = os.path.join(codex_home, "config.toml")
    values = _parse_toml_string_map(config_path)
    root = values.get("", {})
    provider_name = root.get("model_provider")
    if provider_name:
        section = "model_providers.{}".format(provider_name)
        provider = values.get(section, {})
        base_url = provider.get("base_url")
        if base_url:
            return base_url
    custom = values.get("model_providers.custom", {})
    return custom.get("base_url")


def resolve_embedding_settings(codex_home=None, prefer_model=None, prefer_provider=None):
    codex_home = default_codex_home(codex_home)
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or _read_auth_key(codex_home)
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or _read_codex_base_url(codex_home) or DEFAULT_BASE_URL
    model = (prefer_model or os.environ.get("CODEX_MEMORY_EMBED_MODEL") or DEFAULT_EMBED_MODEL).strip()
    provider_mode = (
        prefer_provider
        or os.environ.get("CODEX_MEMORY_EMBED_PROVIDER")
        or "fastembed"
    ).strip()
    dimensions = (os.environ.get("CODEX_MEMORY_EMBED_DIMENSIONS") or "").strip()
    dimensions = int(dimensions) if dimensions else None
    local_model = (
        os.environ.get("CODEX_MEMORY_FASTEMBED_MODEL")
        or DEFAULT_FASTEMBED_MODEL
    ).strip()
    local_cache_dir = (os.environ.get("CODEX_MEMORY_FASTEMBED_CACHE_DIR") or "").strip()
    if not local_cache_dir:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_cache_dir = os.path.join(project_root, ".data", "fastembed-cache")
    return EmbeddingSettings(
        api_key=api_key or "",
        base_url=base_url,
        model=model,
        dimensions=dimensions,
        provider_name=provider_mode,
        local_model=local_model,
        local_cache_dir=local_cache_dir,
    )


def _endpoint_candidates(settings):
    candidates = []
    allow_fallback = (os.environ.get("CODEX_MEMORY_EMBED_ALLOW_OPENAI_FALLBACK") or "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )
    preferred = [settings.base_url]
    if allow_fallback:
        preferred.append(DEFAULT_BASE_URL)
    for base_url in preferred:
        if not base_url:
            continue
        endpoint = base_url.rstrip("/") + "/embeddings"
        if endpoint not in candidates:
            candidates.append(endpoint)
    return candidates


def _request_embeddings(endpoint, settings, batch, timeout):
    body = {
        "model": settings.model,
        "input": batch,
        "encoding_format": "float",
    }
    if settings.dimensions:
        body["dimensions"] = settings.dimensions
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": "Bearer {}".format(settings.api_key),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
    vectors = [row.get("embedding") for row in rows]
    if len(vectors) != len(batch):
        raise EmbeddingError("Embedding response length did not match request length.")
    return vectors


def _embed_texts_remote(texts, settings, batch_size=32, timeout=90):
    if not settings.api_key:
        raise EmbeddingError(
            "No embedding API key found. Set OPENAI_API_KEY or populate ~/.codex/auth.json."
        )
    texts = [text or "" for text in texts]
    last_error = None
    for endpoint in _endpoint_candidates(settings):
        try:
            batched_vectors = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                batched_vectors.extend(_request_embeddings(endpoint, settings, batch, timeout))
            settings.effective_base_url = endpoint.rsplit("/embeddings", 1)[0]
            settings.effective_provider = "openai-compatible"
            settings.effective_model = settings.model
            return batched_vectors
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            last_error = EmbeddingError(
                "Embedding request failed against {}: HTTP {} {}".format(
                    endpoint,
                    exc.code,
                    detail[:300],
                )
            )
        except urllib.error.URLError as exc:
            last_error = EmbeddingError(
                "Embedding request failed against {}: {}".format(endpoint, exc)
            )
        except Exception as exc:
            last_error = EmbeddingError(
                "Embedding request failed against {}: {}".format(endpoint, exc)
            )
    if last_error:
        raise last_error
    raise EmbeddingError(
        "Embedding request failed before reaching any endpoint. "
        "If you want fallback from your configured proxy to api.openai.com, set "
        "CODEX_MEMORY_EMBED_ALLOW_OPENAI_FALLBACK=1."
    )


def _embed_texts_fastembed(texts, settings):
    try:
        from fastembed import TextEmbedding
    except Exception as exc:
        raise EmbeddingError(
            "Local fastembed runtime is not available: {}. Install `fastembed` in the project venv.".format(exc)
        )
    os.makedirs(settings.local_cache_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The model .* now uses mean pooling instead of CLS embedding.*",
        )
        model = TextEmbedding(
            model_name=settings.local_model,
            cache_dir=settings.local_cache_dir,
        )
    vectors = list(model.embed(texts))
    if not vectors:
        raise EmbeddingError("Local fastembed embedding returned no vectors.")
    normalized = []
    for vector in vectors:
        normalized.append([float(value) for value in vector.tolist()])
    settings.effective_provider = "fastembed"
    settings.effective_model = settings.local_model
    settings.effective_base_url = "local://fastembed"
    return normalized


def embed_texts(texts, settings, batch_size=32, timeout=90):
    texts = [text or "" for text in texts]
    mode = (settings.provider_name or "auto").strip().lower()
    last_error = None

    if mode == "fastembed":
        return _embed_texts_fastembed(texts, settings)

    if mode in ("openai", "openai-compatible", "remote"):
        return _embed_texts_remote(texts, settings, batch_size=batch_size, timeout=timeout)

    if mode == "auto":
        try:
            return _embed_texts_remote(texts, settings, batch_size=batch_size, timeout=timeout)
        except EmbeddingError as exc:
            last_error = exc
        return _embed_texts_fastembed(texts, settings)

    raise EmbeddingError(
        "Unknown embedding provider mode: {}. Use auto, fastembed, or openai-compatible.".format(
            settings.provider_name
        )
    )
