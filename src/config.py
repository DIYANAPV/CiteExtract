"""Centralized configuration loader.

Two sources:
  - .env           → secrets and user-specific values (API keys, emails)
  - config.yaml    → application behavior (thresholds, timeouts, models)

All modules import from here instead of reading files or using hardcoded values.
"""

import os
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_ENV_PATH = _PROJECT_ROOT / ".env"

# Singleton: loaded once, reused everywhere
_config: Optional[dict] = None
_env_loaded: bool = False


def _load_env() -> None:
    """Load .env file into os.environ (if it exists).

    Simple parser — no dependency on python-dotenv.
    Supports: KEY=VALUE, KEY="VALUE", and # comments.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True

    if not _ENV_PATH.exists():
        return

    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                # Strip inline comments for unquoted values (e.g. KEY=val # comment)
                comment_idx = value.find(" #")
                if comment_idx > 0:
                    value = value[:comment_idx].strip()
            # Only set if not already in environment (real env vars take precedence)
            if key not in os.environ:
                os.environ[key] = value


def _load() -> dict:
    """Load config from YAML file. Cached after first call."""
    global _config
    if _config is not None:
        return _config
    # Always load .env first so env vars are available
    _load_env()
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at {_CONFIG_PATH}. "
            f"Copy config/config.yaml.example or create one."
        )
    with open(_CONFIG_PATH) as f:
        _config = yaml.safe_load(f) or {}
    return _config


def _get_section(section: str) -> dict:
    """Return a top-level config section as a dict."""
    return _load().get(section, {})


# --- Environment variables (secrets + user-specific) ---

def env(key: str, default: str = "") -> str:
    """Read a value from .env / environment. Loads .env on first call."""
    _load_env()
    return os.environ.get(key, default)


def openai_api_key() -> Optional[str]:
    """OpenAI API key from environment."""
    key = env("OPENAI_API_KEY")
    return key if key else None


def s2_api_key() -> Optional[str]:
    """Semantic Scholar API key from environment."""
    key = env("S2_API_KEY")
    return key if key else None


def serpapi_key() -> Optional[str]:
    """SerpAPI key for Google Scholar search (optional, free tier: 100/month)."""
    key = env("SERPAPI_KEY")
    return key if key else None


def crossref_mailto() -> str:
    """Email for CrossRef polite pool, from environment."""
    return env("CROSSREF_MAILTO", "checkcitation@example.com")


def openalex_mailto() -> str:
    """Email for OpenAlex polite pool, from environment."""
    return env("OPENALEX_MAILTO", "checkcitation@example.com")


# --- GROBID ---

def grobid() -> dict:
    """GROBID service configuration."""
    section = _get_section("grobid")
    retry = section.get("retry", {})
    concurrency = section.get("concurrency", 4)
    return {
        "service_url": os.environ.get("GROBID_SERVICE_URL", section.get("service_url", "http://localhost:8070")),
        "timeout": section.get("timeout", 120),
        "concurrency": concurrency,
        # Default to the L1 pool size when unset so deployments without
        # the new key keep their prior behavior.
        "extract_concurrency": section.get("extract_concurrency", concurrency),
        "health_check_timeout": section.get("health_check_timeout", 5),
        "docker_image": section.get("docker_image", "grobid/grobid:0.8.2-crf"),
        "retry_max_attempts": retry.get("max_attempts", 3),
        "retry_multiplier": retry.get("multiplier", 3),
        "retry_min_wait": retry.get("min_wait", 5),
        "retry_max_wait": retry.get("max_wait", 30),
    }


# --- Parsing ---

def parsing() -> dict:
    """Text parsing / context extraction configuration."""
    section = _get_section("parsing")
    return {
        "context_sentences_before": section.get("context_sentences_before", 2),
        "context_sentences_after": section.get("context_sentences_after", 1),
        "sentence_splitter": section.get("sentence_splitter", "spacy"),
    }


# --- API clients ---

def api(name: str) -> dict:
    """Per-API client configuration (crossref, openalex, semantic_scholar, pubmed)."""
    section = _get_section("apis").get(name, {})
    return section


# --- Rate limits ---

def rate_limits() -> dict[str, tuple[int, int]]:
    """Rate limit configs: {api_name: (max_rate, time_period_seconds)}."""
    section = _get_section("rate_limits")
    defaults = {
        "crossref": (40, 1),
        # S2 free-tier (with or without API key) is 1 req/s. Going higher
        # trips 429 + exponential backoff which is slower than honouring the
        # cap. Match config.yaml's documented value so a missing config file
        # doesn't silently produce 429-storms.
        "semantic_scholar": (1, 1),
        "openalex": (9, 1),
        "pubmed": (2, 1),
    }
    result = {}
    for api_name, default in defaults.items():
        val = section.get(api_name, list(default))
        result[api_name] = tuple(val)
    # Include any extra APIs defined in config
    for api_name, val in section.items():
        if api_name not in result:
            result[api_name] = tuple(val)
    return result


# --- Thresholds ---

def thresholds() -> dict:
    """Matching thresholds."""
    section = _get_section("thresholds")
    return {
        "title_match": section.get("title_match", 0.80),
        "author_match": section.get("author_match", 0.50),
        "venue_match": section.get("venue_match", 0.60),
        "subtitle_ratio": section.get("subtitle_ratio", 0.40),
    }


# --- Risk levels ---

def risk_levels() -> dict:
    """Risk level thresholds for integrity score."""
    section = _get_section("risk_levels")
    return {
        "low": section.get("low", 0.90),
        "medium": section.get("medium", 0.70),
        "high": section.get("high", 0.50),
    }


# --- Cache ---

def cache() -> dict:
    """Cache configuration (paths, TTLs)."""
    section = _get_section("cache")
    return {
        "db_path": section.get("db_path", "data/cache/api_cache.db"),
        "ttl_metadata": section.get("ttl_metadata", 30 * 86400),
        "ttl_abstract": section.get("ttl_abstract", 30 * 86400),
        "ttl_retraction": section.get("ttl_retraction", 7 * 86400),
        "ttl_not_found": section.get("ttl_not_found", 7 * 86400),
        "ttl_fulltext": section.get("ttl_fulltext", 7 * 86400),
    }


# --- Agentic ---

def agentic() -> Optional[dict]:
    """Agentic mode configuration. Returns None if section missing."""
    section = _get_section("agentic")
    if not section:
        return None
    return section


def claim_verification() -> dict:
    """Claim verification LLM configuration."""
    section = _get_section("claim_verification")
    return {
        "provider": "openai",
        "model": section.get("model", "gpt-4o-mini"),
        "temperature": section.get("temperature", 0.0),
        "max_tokens": section.get("max_tokens", 1024),
        "timeout": section.get("timeout", 60),
    }


# --- LLM ---

def llm() -> Optional[dict]:
    """LLM configuration (used by ref parsers and claim verification). Returns None if section missing."""
    section = _get_section("llm")
    if not section:
        return None
    return section


def comprehension() -> dict:
    """Comprehension support configuration (passage retrieval)."""
    section = _get_section("comprehension")
    return {
        "top_k": section.get("top_k", 3),
        "notice_enabled": section.get("notice_enabled", False),
        "dense_model": section.get("dense_model", "all-MiniLM-L6-v2"),
        "bm25_candidates": section.get("bm25_candidates", 10),
        "dense_candidates": section.get("dense_candidates", 10),
        "rrf_k": section.get("rrf_k", 60),
    }


def llm_pricing() -> dict:
    """LLM token pricing (USD per million tokens)."""
    llm_section = _get_section("llm")
    pricing = llm_section.get("pricing", {})
    return {
        "input_cost_per_million": pricing.get("input_cost_per_million", 0.15),
        "output_cost_per_million": pricing.get("output_cost_per_million", 0.60),
    }
