
import os
from typing import Optional

import yaml

from citeextract import paths

_CONFIG_PATH = paths.config_path()
_ENV_PATH = paths.env_path()

_config: Optional[dict] = None
_env_loaded: bool = False


def _load_env() -> None:
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
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                comment_idx = value.find(" #")
                if comment_idx > 0:
                    value = value[:comment_idx].strip()
            if key not in os.environ:
                os.environ[key] = value


def _load() -> dict:
    global _config
    if _config is not None:
        return _config
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
    return _load().get(section, {})


def env(key: str, default: str = "") -> str:
    _load_env()
    return os.environ.get(key, default)


def openai_api_key() -> Optional[str]:
    key = env("OPENAI_API_KEY")
    return key if key else None


def s2_api_key() -> Optional[str]:
    key = env("S2_API_KEY")
    return key if key else None


def serpapi_key() -> Optional[str]:
    key = env("SERPAPI_KEY")
    return key if key else None


def crossref_mailto() -> str:
    return env("CROSSREF_MAILTO", "citeextract@example.com")


def openalex_mailto() -> str:
    return env("OPENALEX_MAILTO", "citeextract@example.com")


def grobid() -> dict:
    section = _get_section("grobid")
    retry = section.get("retry", {})
    concurrency = section.get("concurrency", 4)
    return {
        "service_url": os.environ.get("GROBID_SERVICE_URL", section.get("service_url", "http://localhost:8070")),
        "timeout": section.get("timeout", 120),
        "concurrency": concurrency,
        "extract_concurrency": section.get("extract_concurrency", concurrency),
        "health_check_timeout": section.get("health_check_timeout", 5),
        "docker_image": section.get("docker_image", "grobid/grobid:0.8.2-crf"),
        "retry_max_attempts": retry.get("max_attempts", 3),
        "retry_multiplier": retry.get("multiplier", 3),
        "retry_min_wait": retry.get("min_wait", 5),
        "retry_max_wait": retry.get("max_wait", 30),
    }


def parsing() -> dict:
    section = _get_section("parsing")
    return {
        "context_sentences_before": section.get("context_sentences_before", 2),
        "context_sentences_after": section.get("context_sentences_after", 1),
        "sentence_splitter": section.get("sentence_splitter", "spacy"),
    }


def api(name: str) -> dict:
    section = _get_section("apis").get(name, {})
    return section


def rate_limits() -> dict[str, tuple[int, int]]:
    section = _get_section("rate_limits")
    defaults = {
        "crossref": (40, 1),
        "semantic_scholar": (1, 1),
        "openalex": (9, 1),
        "pubmed": (2, 1),
    }
    result = {}
    for api_name, default in defaults.items():
        val = section.get(api_name, list(default))
        result[api_name] = tuple(val)
    for api_name, val in section.items():
        if api_name not in result:
            result[api_name] = tuple(val)
    return result


def thresholds() -> dict:
    section = _get_section("thresholds")
    return {
        "title_match": section.get("title_match", 0.80),
        "author_match": section.get("author_match", 0.50),
        "venue_match": section.get("venue_match", 0.60),
        "subtitle_ratio": section.get("subtitle_ratio", 0.40),
        "composite_match": section.get("composite_match", 0.65),
    }


def risk_levels() -> dict:
    section = _get_section("risk_levels")
    return {
        "low": section.get("low", 0.90),
        "medium": section.get("medium", 0.70),
        "high": section.get("high", 0.50),
    }


def cache() -> dict:
    section = _get_section("cache")
    return {
        "db_path": section.get("db_path", str(paths.data_dir() / "cache" / "api_cache.db")),
        "ttl_metadata": section.get("ttl_metadata", 30 * 86400),
        "ttl_abstract": section.get("ttl_abstract", 30 * 86400),
        "ttl_retraction": section.get("ttl_retraction", 7 * 86400),
        "ttl_not_found": section.get("ttl_not_found", 7 * 86400),
        "ttl_fulltext": section.get("ttl_fulltext", 7 * 86400),
    }


def experimental_fallbacks() -> dict:
    section = _get_section("experimental_fallbacks") or {}
    return {
        "openreview": bool(section.get("openreview", False)),
    }


def agentic() -> Optional[dict]:
    section = _get_section("agentic")
    if not section:
        return None
    return section


def claim_verification() -> dict:
    section = _get_section("claim_verification")
    return {
        "provider": "openai",
        "model": section.get("model", "gpt-5-mini"),
        "temperature": section.get("temperature", 0.0),
        "max_tokens": section.get("max_tokens", 1024),
        "timeout": section.get("timeout", 60),
    }


def llm() -> Optional[dict]:
    section = _get_section("llm")
    if not section:
        return None
    return section


def comprehension() -> dict:
    section = _get_section("comprehension")
    return {
        "top_k": section.get("top_k", 3),
        "dense_model": section.get("dense_model", "all-MiniLM-L6-v2"),
        "bm25_candidates": section.get("bm25_candidates", 10),
        "dense_candidates": section.get("dense_candidates", 10),
        "rrf_k": section.get("rrf_k", 60),
        "rerank_pool": section.get("rerank_pool", 10),
        "per_ref_fetch_timeout_s": section.get("per_ref_fetch_timeout_s", 30),
    }


def llm_pricing() -> dict:
    llm_section = _get_section("llm")
    pricing = llm_section.get("pricing", {})
    return {
        "input_cost_per_million": pricing.get("input_cost_per_million", 0.15),
        "output_cost_per_million": pricing.get("output_cost_per_million", 0.60),
    }
