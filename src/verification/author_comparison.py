"""Format-aware canonical author comparison for agentic verification.

Replaces fuzzy Jaccard similarity with structured per-author matching.
Understands citation format conventions (Vancouver, APA, IEEE, etc.)
to correctly normalize author names before comparison.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from unidecode import unidecode

from src.citation.format_detector import FORMAT_RULES, get_format_rules


@dataclass
class CanonicalAuthor:
    """A normalized author representation for comparison."""

    surname: str        # lowercase, ASCII, stripped
    first_initial: str  # single lowercase letter, or "" if unknown
    original: str       # the original string for display


@dataclass
class AuthorMatch:
    """Result of matching one reference author against the database."""

    ref_author: str
    canonical_ref: CanonicalAuthor
    db_match: Optional[str]
    canonical_db: Optional[CanonicalAuthor]
    status: str  # "match", "surname_only", "unmatched"


@dataclass
class AuthorComparisonResult:
    """Full comparison result between reference and database author lists."""

    per_author: list[AuthorMatch]
    matched_count: int
    unmatched_count: int
    ref_count: int
    db_count: int
    is_truncated: bool
    truncation_expected: bool
    explanation: str
    format_used: Optional[str]

    def to_dict(self) -> dict:
        """Serialize for the agentic tool response."""
        result: dict = {
            "per_author": [],
            "matched": self.matched_count,
            "unmatched": self.unmatched_count,
            "ref_count": self.ref_count,
            "db_count": self.db_count,
            "is_truncated": self.is_truncated,
            "truncation_expected": self.truncation_expected,
            "explanation": self.explanation,
        }
        if self.format_used:
            result["format_used"] = self.format_used

        for m in self.per_author:
            entry: dict = {
                "ref": m.ref_author,
                "status": m.status,
            }
            if m.db_match:
                entry["db_match"] = m.db_match
            if m.status == "unmatched":
                entry["canonical_surname"] = m.canonical_ref.surname
            result["per_author"].append(entry)

        # Add unmatched authors as a top-level list for easy access
        unmatched = [m.ref_author for m in self.per_author if m.status == "unmatched"]
        if unmatched:
            result["unmatched_authors"] = unmatched

        return result


# ---------------------------------------------------------------------------
# Canonicalization — format-aware author name normalization
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, ASCII-fold, strip punctuation."""
    return re.sub(r'[^a-z ]', '', unidecode(s).lower()).strip()


def _extract_surname_and_initial(name: str) -> tuple[str, str]:
    """Extract (surname, first_initial) from a single author name string.

    Handles common patterns:
        "Smith, John"       → ("smith", "j")
        "Smith, J."         → ("smith", "j")
        "Smith, J. A."      → ("smith", "j")
        "John Smith"        → ("smith", "j")
        "J. Smith"          → ("smith", "j")
        "J. A. Smith"       → ("smith", "j")
        "Smith J"           → ("smith", "j")   (Vancouver)
        "Smith JA"          → ("smith", "j")   (Vancouver)
    """
    name = name.strip()
    if not name:
        return ("", "")

    # Pattern 1: "Last, First..." (APA, Chicago, BibTeX)
    if ',' in name:
        parts = name.split(',', 1)
        surname = _normalize(parts[0])
        rest = parts[1].strip()
        initial = ""
        if rest:
            # Take first letter of the first name/initial
            first_char = re.search(r'[a-zA-Z]', rest)
            if first_char:
                initial = first_char.group(0).lower()
        return (surname, initial)

    # Split by whitespace
    tokens = name.split()
    if len(tokens) == 1:
        return (_normalize(tokens[0]), "")

    # Pattern 2: "J. A. Smith" or "J. Smith" (IEEE) — initials before surname
    # Detect if leading tokens are all initials (single letter, possibly with period)
    initial_tokens = []
    for t in tokens[:-1]:
        clean = t.rstrip('.')
        if len(clean) == 1 and clean.isalpha():
            initial_tokens.append(clean.lower())
        else:
            break

    if initial_tokens and len(initial_tokens) == len(tokens) - 1:
        # All leading tokens are initials → last token is surname
        surname = _normalize(tokens[-1])
        return (surname, initial_tokens[0])

    # Pattern 3: "Smith JA" or "Smith J" (Vancouver) — surname then initials without space
    last_token = tokens[-1]
    if last_token.isupper() and len(last_token) <= 3 and last_token.isalpha():
        surname = _normalize(" ".join(tokens[:-1]))
        return (surname, last_token[0].lower())

    # Pattern 4: "John Smith" or "John Adam Smith" — first names then surname
    surname = _normalize(tokens[-1])
    initial = tokens[0][0].lower() if tokens[0] else ""
    return (surname, initial)


def canonicalize_author(name: str, citation_format: Optional[str] = None) -> CanonicalAuthor:
    """Normalize an author name to canonical form for comparison.

    The citation_format hint can improve accuracy but is not required.
    """
    surname, initial = _extract_surname_and_initial(name)
    return CanonicalAuthor(surname=surname, first_initial=initial, original=name)


# ---------------------------------------------------------------------------
# Comparison — greedy bipartite matching on canonical forms
# ---------------------------------------------------------------------------

def _match_score(a: CanonicalAuthor, b: CanonicalAuthor) -> int:
    """Score how well two canonical authors match.

    Returns:
        2 = surname + initial match
        1 = surname match only (initial missing or different)
        0 = no match
    """
    if a.surname != b.surname:
        return 0
    if a.first_initial and b.first_initial and a.first_initial == b.first_initial:
        return 2
    # Surname matches but initial missing on one side or different
    if not a.first_initial or not b.first_initial:
        return 1  # one side has no initial — accept as surname-only match
    return 0  # different initials with same surname — likely different person


def compare_authors(
    ref_authors: list[str],
    db_authors: list[str],
    citation_format: Optional[str] = None,
) -> AuthorComparisonResult:
    """Compare reference authors against database authors with format awareness.

    Uses canonical (surname, first_initial) matching instead of fuzzy similarity.
    Reports per-author match status and format-aware truncation analysis.
    """
    if not ref_authors:
        return AuthorComparisonResult(
            per_author=[],
            matched_count=0,
            unmatched_count=0,
            ref_count=0,
            db_count=len(db_authors),
            is_truncated=False,
            truncation_expected=False,
            explanation="No authors in reference to compare.",
            format_used=citation_format,
        )

    # Canonicalize both sides
    ref_canonical = [canonicalize_author(a, citation_format) for a in ref_authors]
    db_canonical = [canonicalize_author(a) for a in db_authors]

    # Greedy bipartite matching: for each ref author, find best DB match
    used_db: set[int] = set()
    matches: list[AuthorMatch] = []

    for rc in ref_canonical:
        best_idx = -1
        best_score = 0
        for i, dc in enumerate(db_canonical):
            if i in used_db:
                continue
            s = _match_score(rc, dc)
            if s > best_score:
                best_score = s
                best_idx = i

        if best_score >= 2:
            used_db.add(best_idx)
            matches.append(AuthorMatch(
                ref_author=rc.original,
                canonical_ref=rc,
                db_match=db_canonical[best_idx].original,
                canonical_db=db_canonical[best_idx],
                status="match",
            ))
        elif best_score == 1:
            used_db.add(best_idx)
            matches.append(AuthorMatch(
                ref_author=rc.original,
                canonical_ref=rc,
                db_match=db_canonical[best_idx].original,
                canonical_db=db_canonical[best_idx],
                status="surname_only",
            ))
        else:
            matches.append(AuthorMatch(
                ref_author=rc.original,
                canonical_ref=rc,
                db_match=None,
                canonical_db=None,
                status="unmatched",
            ))

    matched_count = sum(1 for m in matches if m.status in ("match", "surname_only"))
    unmatched_count = sum(1 for m in matches if m.status == "unmatched")
    ref_count = len(ref_authors)
    db_count = len(db_authors)

    # Truncation analysis
    is_truncated = ref_count < db_count
    truncation_expected = False
    fmt_rules = get_format_rules(citation_format)

    if fmt_rules and fmt_rules.et_al_after is not None:
        if db_count > fmt_rules.et_al_after:
            truncation_expected = True

    # Build explanation
    explanation = _build_explanation(
        matches, ref_count, db_count, is_truncated,
        truncation_expected, citation_format, fmt_rules,
    )

    return AuthorComparisonResult(
        per_author=matches,
        matched_count=matched_count,
        unmatched_count=unmatched_count,
        ref_count=ref_count,
        db_count=db_count,
        is_truncated=is_truncated,
        truncation_expected=truncation_expected,
        explanation=explanation,
        format_used=citation_format,
    )


def _build_explanation(
    matches: list[AuthorMatch],
    ref_count: int,
    db_count: int,
    is_truncated: bool,
    truncation_expected: bool,
    citation_format: Optional[str],
    fmt_rules,
) -> str:
    """Build a human-readable explanation of the author comparison."""
    parts: list[str] = []

    matched = sum(1 for m in matches if m.status in ("match", "surname_only"))
    unmatched = sum(1 for m in matches if m.status == "unmatched")

    if matched == ref_count and unmatched == 0:
        parts.append(f"All {ref_count} listed author(s) verified in database record.")
    else:
        parts.append(f"{matched} of {ref_count} listed author(s) matched.")
        if unmatched > 0:
            names = [m.ref_author for m in matches if m.status == "unmatched"]
            parts.append(f"Unmatched: {', '.join(names)}.")

    if is_truncated:
        parts.append(f"Reference lists {ref_count} of {db_count} total authors.")
        if truncation_expected and fmt_rules:
            parts.append(
                f"{fmt_rules.name} format allows truncation after "
                f"{fmt_rules.et_al_after} authors."
            )
        elif citation_format:
            parts.append(
                f"Detected format: {citation_format}. "
                f"Author list appears truncated."
            )
        else:
            parts.append("Author list appears truncated (format not detected).")

    surname_only = [m for m in matches if m.status == "surname_only"]
    if surname_only:
        pairs = [f"'{m.ref_author}' ~ '{m.db_match}'" for m in surname_only]
        parts.append(f"Surname-only matches (initial missing/differs): {', '.join(pairs)}.")

    return " ".join(parts)
