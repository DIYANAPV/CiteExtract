"""Fuzzy matching utilities for comparing reference metadata against DB records.

All API clients use these functions for consistent matching behavior.
"""

import re
from typing import Optional

from rapidfuzz import fuzz
from unidecode import unidecode

from src import config


# --- Web-only sources (shared: pipeline + triage skip metadata for these) ---

WEB_SOURCES = {"web", "web_archive"}


# --- Preprint detection (shared across existence + metadata) ---

PREPRINT_RE = re.compile(
    r'\b(arxiv|biorxiv|medrxiv|ssrn|preprint|e[\s-]?prints?|cornell\s+university'
    r'|chemrxiv|techrxiv|eartharxiv|engrxiv|socarxiv)\b',
    re.IGNORECASE,
)


# --- Title matching ---


def normalize_title(title: str) -> str:
    """Normalize a title for comparison.

    Lowercase, strip punctuation, collapse whitespace, unidecode,
    strip subtitle after colon, remove version markers, normalize dashes/quotes.
    """
    if not title:
        return ""
    t = title.lower().strip()
    # Normalize unicode to ASCII
    t = unidecode(t)
    # Normalize dashes and quotes to ASCII
    t = t.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash
    t = t.replace("\u201c", '"').replace("\u201d", '"')  # smart quotes
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    # Strip subtitle after colon only when the main title is long enough
    # to be meaningful on its own (>= 30 chars) AND the subtitle is very long.
    # Previously this stripped too aggressively, causing titles like
    # "Deepseek-r1: Incentivizing reasoning..." to be reduced to just "deepseek r1".
    colon_match = re.search(r":\s+", t)
    if colon_match:
        main_part = t[:colon_match.start()].strip()
        subtitle = t[colon_match.end():]
        # Only strip if: main part is substantial AND subtitle exceeds threshold
        if len(main_part) >= 30 and len(subtitle) > config.thresholds()["subtitle_ratio"] * len(t):
            t = main_part
    # Remove version markers like "v2", "(v3)"
    t = re.sub(r"\s*\(?v\d+\)?", "", t)
    # Remove punctuation
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(ref_title: str, db_title: str) -> float:
    """Compute normalized similarity between two titles.

    Uses rapidfuzz token_sort_ratio for robustness to word order differences.
    Returns 0.0 - 1.0.
    """
    if not ref_title or not db_title:
        return 0.0
    a = normalize_title(ref_title)
    b = normalize_title(db_title)
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def is_title_match(
    ref_title: Optional[str], db_title: Optional[str], threshold: Optional[float] = None
) -> tuple[bool, float, list[str]]:
    """Check if two titles match and generate flags.

    Returns:
        (is_match, similarity, flags)
        If match but < 1.0, a flag is added for human review.
    """
    if not ref_title or not db_title:
        return False, 0.0, []

    if threshold is None:
        threshold = config.thresholds()["title_match"]

    sim = title_similarity(ref_title, db_title)
    flags: list[str] = []

    if sim < threshold:
        return False, sim, []

    if sim < 1.0:
        flags.append(
            f"title_not_exact_match (similarity={sim:.2f}). "
            f"Found: '{db_title[:80]}'"
        )

    return True, sim, flags


# --- Author matching ---


def normalize_author(name: str) -> str:
    """Normalize an author name for matching.

    Strips hyphens, dots, apostrophes, spaces, converts to lowercase.
    Handles initials vs full names.
    """
    if not name:
        return ""
    n = name.lower().strip()
    n = unidecode(n)
    n = re.sub(r"[-.'\s]", "", n)
    return n


def _is_consortium_name(name: str) -> bool:
    """Detect consortium/group author names that aren't individual people.

    Examples: "the KSS Cave Studies Team", "DeepSeek-AI",
    "The ATLAS Collaboration", "WHO Expert Committee".
    """
    lower = name.lower().strip()
    # Starts with "the " — almost always a consortium
    if lower.startswith("the "):
        return True
    # Known consortium keywords
    consortium_keywords = {
        "team", "collaboration", "consortium", "committee", "group",
        "network", "initiative", "project", "working party", "taskforce",
    }
    for kw in consortium_keywords:
        if kw in lower:
            return True
    # All uppercase or single token with dash (e.g., "DeepSeek-AI", "OpenAI")
    if '-' in name and len(name.split()) == 1 and name[0].isupper():
        # Could be "DeepSeek-AI" — check if it's not a hyphenated surname
        parts = name.split('-')
        if any(p.isupper() for p in parts):
            return True
    return False


def _author_tokens(authors: list[str]) -> set[str]:
    """Extract normalized surname tokens from a list of author names.

    Handles both "First Last" and "Last, First" formats.
    Skips consortium/group author names (e.g., "the KSS Cave Studies Team").
    """
    tokens: set[str] = set()
    for name in authors:
        name = name.strip()
        if not name:
            continue
        if _is_consortium_name(name):
            continue
        if ',' in name:
            # "Last, First" or "Last, F." — surname is before the comma
            surname = name.split(',')[0].strip()
        else:
            # "First Last" or "First M. Last" — surname is the last word
            parts = name.split()
            surname = parts[-1] if parts else name
        token = normalize_author(surname)
        if len(token) > 1:  # skip single-letter initials
            tokens.add(token)
    return tokens


def author_similarity(ref_authors: list[str], db_authors: list[str]) -> float:
    """Jaccard similarity over normalized author surname tokens.

    Returns 0.0 - 1.0.
    """
    if not ref_authors or not db_authors:
        return 0.0
    ref_tokens = _author_tokens(ref_authors)
    db_tokens = _author_tokens(db_authors)
    if not ref_tokens or not db_tokens:
        return 0.0
    intersection = ref_tokens & db_tokens
    union = ref_tokens | db_tokens
    return len(intersection) / len(union)


def is_author_truncation(ref_authors: list[str], db_authors: list[str]) -> bool:
    """Detect if the reference author list is a truncated subset of the DB list.

    BibTeX entries routinely list only the first 3-5 authors with "et al."
    for papers with 10-100+ authors. This causes low Jaccard scores even
    when all listed authors are correct.

    A truncation pattern requires ALL of:
    1. DB has significantly more authors (> 2x the ref list)
    2. Nearly all ref authors appear in the DB (containment >= 0.80)

    The 0.80 containment threshold allows at most 1 name variation per 5
    authors (e.g., transliteration differences), but blocks blended refs
    where multiple listed authors are from a different paper.

    Returns True only when the low Jaccard score is explained by truncation
    rather than fabrication.
    """
    if not ref_authors or not db_authors:
        return False
    ref_tokens = _author_tokens(ref_authors)
    db_tokens = _author_tokens(db_authors)
    if not ref_tokens or not db_tokens:
        return False

    # Condition 1: DB list must be significantly larger.
    # At ratio 2x, Jaccard for a perfect subset = ref/(2*ref) = 0.50
    # which is right at the threshold. One name variation would fail it,
    # so we include the boundary (>=) in truncation detection.
    if len(db_tokens) < len(ref_tokens) * 2:
        return False

    # Condition 2: Nearly all ref authors must be in the DB
    intersection = ref_tokens & db_tokens
    containment = len(intersection) / len(ref_tokens)
    return containment >= 0.80


# --- Year comparison (context-aware: preprint vs publication dates) ---


def compare_year(
    ref_year: Optional[int], db_year: Optional[int]
) -> dict:
    """Compare years with awareness of preprint-vs-publication date patterns.

    Academic papers commonly have different dates across sources:
    - arXiv preprint date (often 1 year before conference proceedings)
    - Conference/journal publication date
    - Online-first date

    A 1-year difference is a systematic, predictable pattern in CS/ML
    (e.g. arXiv 2015, CVPR 2016 for the same paper). This is not an error
    but a legitimate date discrepancy across sources.

    Returns:
        {
            'match': bool,
            'close_match': bool,   # True if years differ by exactly 1
            'ref_year': ...,
            'db_year': ...,
            'flag': Optional[str],
        }
    """
    if ref_year is None or db_year is None:
        return {
            "match": True,
            "close_match": False,
            "ref_year": ref_year,
            "db_year": db_year,
            "flag": None,
        }

    if ref_year == db_year:
        return {
            "match": True,
            "close_match": False,
            "ref_year": ref_year,
            "db_year": db_year,
            "flag": None,
        }

    diff = abs(ref_year - db_year)

    if diff == 1:
        return {
            "match": True,
            "close_match": True,
            "ref_year": ref_year,
            "db_year": db_year,
            "flag": (
                f"year_close_match: paper says {ref_year}, database says {db_year} "
                f"(likely preprint-vs-publication date difference)"
            ),
        }

    return {
        "match": False,
        "close_match": False,
        "ref_year": ref_year,
        "db_year": db_year,
        "flag": f"year_mismatch: paper says {ref_year}, database says {db_year}",
    }
