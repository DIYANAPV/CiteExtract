
import re
from typing import Optional

from rapidfuzz import fuzz
from unidecode import unidecode

from citeextract import config


WEB_SOURCES = {"web", "web_archive"}


PREPRINT_RE = re.compile(
    r'\b(arxiv|biorxiv|medrxiv|ssrn|preprint|e[\s-]?prints?|cornell\s+university'
    r'|chemrxiv|techrxiv|eartharxiv|engrxiv|socarxiv)\b',
    re.IGNORECASE,
)


def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower().strip()
    t = unidecode(t)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    colon_match = re.search(r":\s+", t)
    if colon_match:
        main_part = t[:colon_match.start()].strip()
        subtitle = t[colon_match.end():]
        if len(main_part) >= 30 and len(subtitle) > config.thresholds()["subtitle_ratio"] * len(t):
            t = main_part
    t = re.sub(r"\s*\(?v\d+\)?", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(ref_title: str, db_title: str) -> float:
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


def normalize_author(name: str) -> str:
    if not name:
        return ""
    n = name.lower().strip()
    n = unidecode(n)
    n = re.sub(r"[-.'\s]", "", n)
    return n


_CORPORATE_AUTHOR_NAMES = frozenset({
    "openai", "anthropic", "deepmind", "google deepmind", "meta ai",
    "google ai", "google research", "google brain", "facebook ai research",
    "fair", "microsoft research", "ibm research", "nvidia research",
    "apple machine learning research", "apple", "amazon science",
    "salesforce research", "huggingface", "hugging face", "cohere",
    "allen institute for ai", "allen institute for artificial intelligence",
    "ai2", "stability ai", "mistral ai", "databricks", "snowflake ai research",
    "deepseek ai", "deepseek-ai", "qwen team", "ollama",
})

_CORPORATE_TRAILING_TOKENS = frozenset({
    "ai", "research", "lab", "labs", "team", "inc", "ltd", "corp",
    "foundation", "group", "institute",
})


def _is_consortium_name(name: str) -> bool:
    raw = name.strip()
    if not raw:
        return False
    lower = raw.lower()

    if lower in _CORPORATE_AUTHOR_NAMES:
        return True

    if lower.startswith("the "):
        return True

    consortium_keywords = {
        "team", "collaboration", "consortium", "committee", "group",
        "network", "initiative", "project", "working party", "taskforce",
    }
    for kw in consortium_keywords:
        if kw in lower:
            return True

    tokens = raw.split()
    if 1 <= len(tokens) <= 4:
        last = tokens[-1].lower().rstrip(".,;:")
        if last in _CORPORATE_TRAILING_TOKENS:
            return True

    if '-' in raw and len(tokens) == 1 and raw[0].isupper():
        parts = raw.split('-')
        if any(p.isupper() for p in parts):
            return True

    return False


def _author_tokens(authors: list[str]) -> set[str]:
    tokens: set[str] = set()
    for name in authors:
        name = name.strip()
        if not name:
            continue
        if _is_consortium_name(name):
            continue
        if ',' in name:
            surname = name.split(',')[0].strip()
        else:
            parts = name.split()
            surname = parts[-1] if parts else name
        token = normalize_author(surname)
        if len(token) > 1:
            tokens.add(token)
    return tokens


def author_similarity(ref_authors: list[str], db_authors: list[str]) -> float:
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
    if not ref_authors or not db_authors:
        return 0.0
    ref_tokens = _author_tokens(ref_authors)
    db_tokens = _author_tokens(db_authors)
    if not ref_tokens or not db_tokens:
        return 0.0
    return len(ref_tokens & db_tokens) / len(ref_tokens)


def is_author_truncation(ref_authors: list[str], db_authors: list[str]) -> bool:
    if not ref_authors or not db_authors:
        return False
    ref_tokens = _author_tokens(ref_authors)
    db_tokens = _author_tokens(db_authors)
    if not ref_tokens or not db_tokens:
        return False

    if len(db_tokens) < len(ref_tokens) * 2:
        return False

    intersection = ref_tokens & db_tokens
    containment = len(intersection) / len(ref_tokens)
    return containment >= 0.80


_W_TITLE = 0.5
_W_AUTHOR = 0.3
_W_YEAR = 0.2

COMPOSITE_MATCH_THRESHOLD = 0.65


def _year_score(ref_year: Optional[int], cand_year: Optional[int]) -> float:
    if ref_year is None or cand_year is None:
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
    if not ref_authors or not cand_authors:
        return False
    return author_similarity(ref_authors, cand_authors) == 0.0


MATCH_STRATEGY_COMPOSITE = "composite"
MATCH_STRATEGY_TITLE_ONLY = "title_only"


def pick_best_candidate(
    ref_title: Optional[str], ref_authors: list[str], ref_year: Optional[int],
    candidates: list[dict], title_threshold: Optional[float] = None,
    composite_threshold: Optional[float] = None,
) -> tuple[Optional[dict], str, float]:
    cfg = config.thresholds()
    if title_threshold is None:
        title_threshold = cfg["title_match"]
    if composite_threshold is None:
        composite_threshold = cfg.get("composite_match", COMPOSITE_MATCH_THRESHOLD)

    best_composite: Optional[tuple[float, dict]] = None
    for cand in candidates:
        cand_authors = cand.get("authors") or []
        if is_authors_disjoint(ref_authors or [], cand_authors):
            continue
        score = composite_match_score(
            ref_title, ref_authors or [], ref_year,
            cand.get("title"), cand_authors, cand.get("year"),
        )
        if score >= composite_threshold:
            if best_composite is None or score > best_composite[0]:
                best_composite = (score, cand)

    if best_composite is not None:
        return best_composite[1], MATCH_STRATEGY_COMPOSITE, best_composite[0]

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


ID_SYSTEM_DOI = "doi"
ID_SYSTEM_ARXIV = "arxiv"
ID_SYSTEM_ACL = "acl"


_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_ARXIV_URL_RE = re.compile(
    r"(?:https?://)?arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_ARXIV_DOI_RE = re.compile(
    r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.IGNORECASE,
)

_ACL_ID_RE = re.compile(
    r"\b([A-Z]\d{2}-\d{4}|\d{4}\.[a-z]+(?:-[a-z]+)?\.\d+)\b",
    re.IGNORECASE,
)
_ACL_URL_RE = re.compile(
    r"(?:https?://)?aclanthology\.org/"
    r"([A-Z]\d{2}-\d{4}|\d{4}\.[a-z]+(?:-[a-z]+)?\.\d+)/?",
    re.IGNORECASE,
)
_ACL_DOI_RE = re.compile(
    r"10\.18653/v\d+/([A-Z]\d{2}-\d{4}|\d{4}\.[a-z]+(?:-[a-z]+)?\.\d+)",
    re.IGNORECASE,
)

_DOI_RE = re.compile(r"10\.\d{4,9}/[\w\.\-/;:()<>]+", re.IGNORECASE)


def canonical_id(raw: str) -> Optional[tuple[str, str]]:
    if not raw:
        return None
    s = raw.strip().rstrip(".,;)/").lstrip("(")
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", s, flags=re.IGNORECASE)

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

    m = _ARXIV_ID_RE.fullmatch(s)
    if m:
        return (ID_SYSTEM_ARXIV, m.group(1).lower())

    m = _ACL_ID_RE.fullmatch(s)
    if m:
        return (ID_SYSTEM_ACL, m.group(1).lower())

    return None


def ids_equivalent(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    ca = canonical_id(a)
    cb = canonical_id(b)
    return ca is not None and ca == cb


_DB_ID_FIELDS = ("doi", "arxiv_id", "anthology_id", "acl_id")


def matches_any_known_id(
    user_id: Optional[str], db_record: dict,
) -> bool:
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


def compare_year(
    ref_year: Optional[int], db_year: Optional[int],
    *, strong_match: bool = False,
) -> dict:
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

    if diff == 2 and strong_match:
        return {
            "match": True,
            "close_match": True,
            "ref_year": ref_year,
            "db_year": db_year,
            "flag": (
                f"year_close_match: paper says {ref_year}, database says {db_year} "
                f"(2-year gap accepted because title and authors agree — "
                f"likely versioned arXiv repost or slow journal pipeline)"
            ),
        }

    return {
        "match": False,
        "close_match": False,
        "ref_year": ref_year,
        "db_year": db_year,
        "flag": f"year_mismatch: paper says {ref_year}, database says {db_year}",
    }
