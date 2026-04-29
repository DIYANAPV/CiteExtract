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


_CORPORATE_AUTHOR_NAMES = frozenset({
    # Org-only labels we see verbatim in bibliographies.
    "openai", "anthropic", "deepmind", "google deepmind", "meta ai",
    "google ai", "google research", "google brain", "facebook ai research",
    "fair", "microsoft research", "ibm research", "nvidia research",
    "apple machine learning research", "apple", "amazon science",
    "salesforce research", "huggingface", "hugging face", "cohere",
    "allen institute for ai", "allen institute for artificial intelligence",
    "ai2", "stability ai", "mistral ai", "databricks", "snowflake ai research",
    "deepseek ai", "deepseek-ai", "qwen team", "ollama",
})

# Trailing tokens that mark a name as a corporate/research org rather than a
# person. Matched on the *last* whitespace-separated token (case-insensitive)
# so "Meta AI" → trailing "ai", "Google Research" → trailing "research".
_CORPORATE_TRAILING_TOKENS = frozenset({
    "ai", "research", "lab", "labs", "team", "inc", "ltd", "corp",
    "foundation", "group", "institute",
})


def _is_consortium_name(name: str) -> bool:
    """Detect consortium/group author names that aren't individual people.

    Catches three patterns:

    1. **Stylistic markers** — "the …", "X Collaboration", "Y Consortium"
       (the historical case the existing test suite covers).
    2. **Known corporate authors** — exact matches against a curated list
       (Meta AI, Google Research, Anthropic, OpenAI …) so the cross-
       validation guard in :func:`_cross_validate_authors` doesn't flag
       e.g. "Meta AI" as a fabricated co-author on a real Llama paper.
    3. **Org-suffix names** — anything ending in ``AI`` / ``Research`` /
       ``Lab`` / ``Inc`` etc., which catches the long tail of one-off
       corporate labels we don't enumerate explicitly.

    Examples: "the KSS Cave Studies Team", "DeepSeek-AI",
    "The ATLAS Collaboration", "WHO Expert Committee", "Meta AI",
    "Google Research".
    """
    raw = name.strip()
    if not raw:
        return False
    lower = raw.lower()

    # 2. Known corporate-author full names.
    if lower in _CORPORATE_AUTHOR_NAMES:
        return True

    # 1a. Starts with "the " — almost always a consortium
    if lower.startswith("the "):
        return True

    # 1b. Known consortium keywords anywhere in the name
    consortium_keywords = {
        "team", "collaboration", "consortium", "committee", "group",
        "network", "initiative", "project", "working party", "taskforce",
    }
    for kw in consortium_keywords:
        if kw in lower:
            return True

    # 3. Org-suffix names: trailing token is a corporate marker.
    # Limited to short author strings (≤ 4 tokens) to avoid false positives
    # on "Andrew Y. Ng Lab Director" style nonsense; real corporate authors
    # are short — "Meta AI", "Google Research", "Allen AI Institute".
    tokens = raw.split()
    if 1 <= len(tokens) <= 4:
        last = tokens[-1].lower().rstrip(".,;:")
        if last in _CORPORATE_TRAILING_TOKENS:
            return True

    # 1c. Hyphenated single-token uppercase forms (e.g., "DeepSeek-AI").
    if '-' in raw and len(tokens) == 1 and raw[0].isupper():
        parts = raw.split('-')
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


def author_containment(ref_authors: list[str], db_authors: list[str]) -> float:
    """Fraction of *reference* author surnames that appear in the DB record.

    Less symmetric than :func:`author_similarity`. Used by the title-evolved
    arXiv rescue path: we want a high score when every author the user
    listed shows up in the candidate, regardless of how many additional
    authors the candidate has. (arXiv records often add late-stage
    co-authors not in the user's bibliography copy.)

    Returns 0.0 - 1.0; 0 when either side is empty after consortium /
    corporate-author filtering.
    """
    if not ref_authors or not db_authors:
        return 0.0
    ref_tokens = _author_tokens(ref_authors)
    db_tokens = _author_tokens(db_authors)
    if not ref_tokens or not db_tokens:
        return 0.0
    return len(ref_tokens & db_tokens) / len(ref_tokens)


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


# ---------------------------------------------------------------------------
# Composite candidate scoring
# ---------------------------------------------------------------------------
#
# Single-axis "best title similarity wins" picks the wrong paper when titles
# share vocabulary. Two real examples we hit:
#
#   ref:  "Natural Language Processing with Python" (Bird/Klein/Loper, 2009)
#   pick: "Python for Natural Language Processing"  (Sarkar, 2019)
#         token_sort_ratio ≈ 0.88 → "matched", but it's a different book.
#
#   ref:  "The Semantic Web" (Berners-Lee, 2001)
#   pick: a 2011 Lecture Notes book of the same name (Goos/Hartmanis et al.)
#         identical normalized title → matched, but completely different paper.
#
# The discriminator in both cases is **authors** (and to a lesser extent
# year). Composite scoring blends all three so a high title score that
# contradicts authors gets dragged down below the accept bar.


# Weights for the composite. Title still dominates (it's the user's
# primary search anchor) but authors carry meaningful weight, and year
# breaks ties between same-title-different-paper candidates.
_W_TITLE = 0.5
_W_AUTHOR = 0.3
_W_YEAR = 0.2

# Composite score required to accept a candidate. Calibrated so that
# title >= 0.80 + authors-disjoint never reaches the bar (max would be
# 0.5*0.80 + 0.3*0 + 0.2*1 = 0.60 — and we additionally hard-reject the
# disjoint-author case).
COMPOSITE_MATCH_THRESHOLD = 0.65


def _year_score(ref_year: Optional[int], cand_year: Optional[int]) -> float:
    """Year similarity in [0, 1]. Lenient about ±1 (preprint vs proceedings)
    and ±2 (versioned arXiv reposts), strict about larger gaps."""
    if ref_year is None or cand_year is None:
        # Missing on either side — neutral signal so we don't penalize.
        return 0.5
    diff = abs(ref_year - cand_year)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.8
    if diff <= 2:
        return 0.5
    if diff <= 5:
        return 0.2
    return 0.0


def composite_match_score(
    ref_title: Optional[str], ref_authors: list[str], ref_year: Optional[int],
    cand_title: Optional[str], cand_authors: list[str], cand_year: Optional[int],
) -> float:
    """Weighted blend of title + author + year similarity.

    Returns a score in ``[0, 1]``. Used by DB clients to pick the best
    candidate from a result list when more than one share a similar
    title — without authors as a tiebreaker, the picker grabs the
    wrong paper for vocabulary-overlapping titles.

    Authors and year contribute neutrally (0.5) when missing on one
    side, so refs without explicit author lists don't get penalized
    relative to those that have them.
    """
    title_s = title_similarity(ref_title or "", cand_title or "") if (
        ref_title and cand_title
    ) else 0.0
    if ref_authors and cand_authors:
        author_s = author_similarity(ref_authors, cand_authors)
    else:
        author_s = 0.5
    year_s = _year_score(ref_year, cand_year)
    return _W_TITLE * title_s + _W_AUTHOR * author_s + _W_YEAR * year_s


def is_authors_disjoint(
    ref_authors: list[str], cand_authors: list[str],
) -> bool:
    """Hard-reject signal: both sides have authors, zero surnames overlap.

    Two papers can't share zero authors AND be the same paper. When
    this holds we ignore title similarity entirely — title alone is
    not a strong enough signal to overrule complete author divergence.
    """
    if not ref_authors or not cand_authors:
        return False
    return author_similarity(ref_authors, cand_authors) == 0.0


# Match strategy labels used in returned records so downstream layers
# can flag low-author-confidence matches in the verdict / UI.
MATCH_STRATEGY_COMPOSITE = "composite"   # title + authors + year all aligned
MATCH_STRATEGY_TITLE_ONLY = "title_only"  # title-only fallback (authors hallucinated?)


def pick_best_candidate(
    ref_title: Optional[str], ref_authors: list[str], ref_year: Optional[int],
    candidates: list[dict], title_threshold: Optional[float] = None,
) -> tuple[Optional[dict], str, float]:
    """Pick the best candidate using the two-pass strategy.

    Pass 1 (composite): score every candidate by title+authors+year.
    Reject any with disjoint author lists (hard signal of "different
    paper that happens to share words"). Best composite score >=
    :data:`COMPOSITE_MATCH_THRESHOLD` wins.

    Pass 2 (title-only): if Pass 1 found nothing, fall back to the
    classical title-only pick — but flag the result as
    ``MATCH_STRATEGY_TITLE_ONLY`` so callers can mark the match as
    low-author-confidence in the verdict.

    The fallback exists because the citing paper's authors might be
    *hallucinated* — a fabricated citation may attach made-up names to
    a real-paper title. Refusing the title-only match would mean we
    can't even surface the wrong-author finding.

    Args:
        ref_title / ref_authors / ref_year: the local Reference's fields.
        candidates: list of DB-record dicts each with "title",
            "authors", "year" keys.
        title_threshold: minimum title similarity for the fallback pass.
            Defaults to ``config.thresholds()["title_match"]``.

    Returns:
        ``(candidate_or_None, match_strategy, similarity_score)``.
        ``similarity_score`` is the composite score for Pass 1 wins,
        the title similarity for Pass 2 wins, or 0.0 when nothing
        cleared either bar.
    """
    if title_threshold is None:
        title_threshold = config.thresholds()["title_match"]

    # --- Pass 1: composite scoring with author-disjoint hard reject ---
    best_composite: Optional[tuple[float, dict]] = None
    for cand in candidates:
        cand_authors = cand.get("authors") or []
        if is_authors_disjoint(ref_authors or [], cand_authors):
            continue
        score = composite_match_score(
            ref_title, ref_authors or [], ref_year,
            cand.get("title"), cand_authors, cand.get("year"),
        )
        if score >= COMPOSITE_MATCH_THRESHOLD:
            if best_composite is None or score > best_composite[0]:
                best_composite = (score, cand)

    if best_composite is not None:
        return best_composite[1], MATCH_STRATEGY_COMPOSITE, best_composite[0]

    # --- Pass 2: title-only fallback (hallucinated-author tolerant) ---
    best_title: Optional[tuple[float, dict]] = None
    for cand in candidates:
        sim = title_similarity(ref_title or "", cand.get("title") or "")
        if sim < title_threshold:
            continue
        if best_title is None or sim > best_title[0]:
            best_title = (sim, cand)

    if best_title is not None:
        return best_title[1], MATCH_STRATEGY_TITLE_ONLY, best_title[0]

    return None, MATCH_STRATEGY_TITLE_ONLY, 0.0


# ---------------------------------------------------------------------------
# Canonical paper identifiers
# ---------------------------------------------------------------------------
#
# Same paper, multiple identifier shapes the user might cite:
#
#     URL form           bare form            cross-system DOI
#     -------------      -----------------    -----------------------------
#     doi.org/10.X/Y     10.X/Y               (DOI is canonical)
#     arxiv.org/abs/Z    Z (e.g. 2407.21783)  10.48550/arxiv.Z
#     aclanthology.org/X X (e.g. Q16-1026)    10.18653/v1/X
#
# The metadata-comparison layer needs to recognize that a citation
# carrying ``aclanthology.org/q16-1026`` and a DB record holding
# DOI ``10.18653/v1/Q16-1026`` refer to the same paper. Without that,
# users see "DOI mismatch" verdicts on entirely correct citations.
#
# ``canonical_id`` parses a raw identifier string into a ``(system, value)``
# pair where ``value`` is normalized (lowercased, stripped of URL prefix
# and ``vN`` suffix). Two identifiers refer to the same paper iff their
# canonical forms are equal, OR if cross-record matching finds the
# user's identifier in any of the DB's other identifier fields.


# Identifier system labels surfaced by ``canonical_id``. Stable enums
# so callers can pattern-match without parsing the string.
ID_SYSTEM_DOI = "doi"
ID_SYSTEM_ARXIV = "arxiv"
ID_SYSTEM_ACL = "acl"


# arXiv ID — modern format (4 digits + "." + 4-5 digits, optional vN suffix)
_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
# arXiv URL: https://arxiv.org/abs/2407.21783[v2] or .../pdf/2407.21783[v2]
_ARXIV_URL_RE = re.compile(
    r"(?:https?://)?arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
# arXiv DOI: 10.48550/arxiv.<id>
_ARXIV_DOI_RE = re.compile(
    r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.IGNORECASE,
)

# ACL Anthology — both legacy ("Q16-1026") and modern
# ("2020.acl-main.123") IDs.
_ACL_ID_RE = re.compile(
    r"\b([A-Z]\d{2}-\d{4}|\d{4}\.[a-z]+(?:-[a-z]+)?\.\d+)\b",
    re.IGNORECASE,
)
_ACL_URL_RE = re.compile(
    r"(?:https?://)?aclanthology\.org/"
    r"([A-Z]\d{2}-\d{4}|\d{4}\.[a-z]+(?:-[a-z]+)?\.\d+)/?",
    re.IGNORECASE,
)
# ACL DOI: 10.18653/v1/<acl-id>  (used by ACL since 2017 or so)
_ACL_DOI_RE = re.compile(
    r"10\.18653/v\d+/([A-Z]\d{2}-\d{4}|\d{4}\.[a-z]+(?:-[a-z]+)?\.\d+)",
    re.IGNORECASE,
)

# Generic DOI (last-ditch — must come AFTER the system-specific patterns
# above, otherwise ``10.48550/arxiv.X`` would canonicalize as a DOI).
_DOI_RE = re.compile(r"10\.\d{4,9}/[\w\.\-/;:()<>]+", re.IGNORECASE)


def canonical_id(raw: str) -> Optional[tuple[str, str]]:
    """Parse an identifier into ``(system, normalized_value)`` or ``None``.

    Recognizes URLs, bare IDs, ``doi:`` prefix, arXiv DOIs, and ACL DOIs.
    Lowercases and strips version suffixes so two equivalent identifiers
    return the exact same tuple.

    Examples:
        >>> canonical_id("https://doi.org/10.1234/foo")
        ('doi', '10.1234/foo')
        >>> canonical_id("arXiv:2407.21783v2")
        ('arxiv', '2407.21783')
        >>> canonical_id("10.48550/arxiv.2407.21783")
        ('arxiv', '2407.21783')
        >>> canonical_id("aclanthology.org/Q16-1026")
        ('acl', 'q16-1026')
        >>> canonical_id("10.18653/v1/Q16-1026")
        ('acl', 'q16-1026')
    """
    if not raw:
        return None
    s = raw.strip().rstrip(".,;)/").lstrip("(")
    # Strip known URL/scheme + ``doi:`` prefixes before pattern matching.
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", s, flags=re.IGNORECASE)

    # Order matters: arXiv-DOI before generic DOI; ACL-DOI before generic DOI.
    m = _ARXIV_DOI_RE.search(s)
    if m:
        return (ID_SYSTEM_ARXIV, m.group(1).lower())

    m = _ACL_DOI_RE.search(s)
    if m:
        return (ID_SYSTEM_ACL, m.group(1).lower())

    m = _ARXIV_URL_RE.search(s)
    if m:
        return (ID_SYSTEM_ARXIV, m.group(1).lower())

    m = _ACL_URL_RE.search(s)
    if m:
        return (ID_SYSTEM_ACL, m.group(1).lower())

    m = _DOI_RE.search(s)
    if m:
        return (ID_SYSTEM_DOI, m.group().lower().rstrip(".,;)"))

    # Bare arXiv ID and ACL ID (no scheme, no prefix). ArXiv is checked
    # before ACL because the ID shapes don't overlap; ordering is for
    # readability.
    m = _ARXIV_ID_RE.fullmatch(s)
    if m:
        return (ID_SYSTEM_ARXIV, m.group(1).lower())

    m = _ACL_ID_RE.fullmatch(s)
    if m:
        return (ID_SYSTEM_ACL, m.group(1).lower())

    return None


def ids_equivalent(a: Optional[str], b: Optional[str]) -> bool:
    """True iff two raw identifier strings refer to the same paper.

    Equivalent means: canonicalize both, compare the resulting
    ``(system, value)`` tuples. Returns ``False`` when either input is
    empty or unrecognizable — never raises.
    """
    if not a or not b:
        return False
    ca = canonical_id(a)
    cb = canonical_id(b)
    return ca is not None and ca == cb


# Fields on a DB record (or ExistenceResult) where alternate identifiers
# may live. Order matters only for log-readability; lookup is set-like.
_DB_ID_FIELDS = ("doi", "arxiv_id", "anthology_id", "acl_id")


def matches_any_known_id(
    user_id: Optional[str], db_record: dict,
) -> bool:
    """True iff ``user_id`` matches ANY identifier the DB has for the paper.

    Checks every field in ``_DB_ID_FIELDS`` (DOI, arXiv ID, ACL ID, …).
    Lets us recognize that ``aclanthology.org/q16-1026`` (ref) and
    ``10.1162/tacla00104`` + ``Q16-1026`` (DB record with both DOI and
    anthology ID stored) refer to the same paper, even though the DOI
    strings differ — they share the ACL anthology key when canonicalized.
    """
    if not user_id:
        return False
    user = canonical_id(user_id)
    if user is None:
        return False
    for field in _DB_ID_FIELDS:
        v = db_record.get(field) if isinstance(db_record, dict) else getattr(db_record, field, None)
        if not v:
            continue
        if canonical_id(v) == user:
            return True
    return False


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
