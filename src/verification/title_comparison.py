"""Structured title comparison for agentic verification.

Replaces single-number fuzzy similarity with a word-level diff that shows
exactly what differs between reference and database titles.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz
from unidecode import unidecode


@dataclass
class TitleComparisonResult:
    """Full comparison result between reference and database titles."""

    exact_match: bool           # True if normalized forms are identical
    normalized_ref: str         # normalized reference title
    normalized_db: str          # normalized database title
    differences: list[str]      # human-readable list of what differs
    similarity: float           # 0.0-1.0 (for backward compat / reference)

    def to_dict(self) -> dict:
        """Serialize for the agentic tool response."""
        result: dict = {
            "exact_match": self.exact_match,
            "normalized_ref": self.normalized_ref,
            "normalized_db": self.normalized_db,
            "similarity": round(self.similarity, 4),
        }
        if self.differences:
            result["differences"] = self.differences
        return result


# ---------------------------------------------------------------------------
# Normalization — same logic as matching.py but without config dependency
# ---------------------------------------------------------------------------

def _normalize(title: str) -> str:
    """Normalize a title for comparison."""
    if not title:
        return ""
    t = title.lower().strip()
    t = unidecode(t)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = re.sub(r"\s*\(?v\d+\)?", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Word-level diff
# ---------------------------------------------------------------------------

def _word_diff(a_words: list[str], b_words: list[str]) -> list[str]:
    """Produce human-readable differences between two word lists.

    Uses a simple LCS-based approach to identify added, removed, and changed words.
    """
    differences: list[str] = []

    # Build longest common subsequence table
    m, n = len(a_words), len(b_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_words[i - 1] == b_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find differences
    i, j = m, n
    ref_only: list[str] = []
    db_only: list[str] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and a_words[i - 1] == b_words[j - 1]:
            # Flush accumulated differences
            if ref_only or db_only:
                differences.append(_format_diff(ref_only[::-1], db_only[::-1]))
                ref_only, db_only = [], []
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            db_only.append(b_words[j - 1])
            j -= 1
        else:
            ref_only.append(a_words[i - 1])
            i -= 1

    if ref_only or db_only:
        differences.append(_format_diff(ref_only[::-1], db_only[::-1]))

    return differences


def _format_diff(ref_words: list[str], db_words: list[str]) -> str:
    """Format a single difference between ref and db."""
    ref_str = " ".join(ref_words) if ref_words else None
    db_str = " ".join(db_words) if db_words else None

    if ref_str and db_str:
        return f"ref has '{ref_str}' where db has '{db_str}'"
    elif ref_str:
        return f"ref has extra word(s): '{ref_str}'"
    elif db_str:
        return f"db has extra word(s): '{db_str}'"
    return ""


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_titles(ref_title: str, db_title: str) -> TitleComparisonResult:
    """Compare reference title against database title with word-level diff.

    Returns structured result showing exact match status and specific differences.
    """
    if not ref_title or not db_title:
        return TitleComparisonResult(
            exact_match=False,
            normalized_ref=_normalize(ref_title or ""),
            normalized_db=_normalize(db_title or ""),
            differences=["One or both titles are empty."],
            similarity=0.0,
        )

    norm_ref = _normalize(ref_title)
    norm_db = _normalize(db_title)

    # Compute similarity for backward compat
    similarity = fuzz.token_sort_ratio(norm_ref, norm_db) / 100.0 if norm_ref and norm_db else 0.0

    if norm_ref == norm_db:
        return TitleComparisonResult(
            exact_match=True,
            normalized_ref=norm_ref,
            normalized_db=norm_db,
            differences=[],
            similarity=similarity,
        )

    # Word-level diff
    ref_words = norm_ref.split()
    db_words = norm_db.split()
    differences = _word_diff(ref_words, db_words)

    return TitleComparisonResult(
        exact_match=False,
        normalized_ref=norm_ref,
        normalized_db=norm_db,
        differences=differences,
        similarity=similarity,
    )
