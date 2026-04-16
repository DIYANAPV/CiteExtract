"""Citation filters — shared utilities for filtering trivial citations."""

import re

from src.models.citation import Citation


_TRIVIAL_PATTERNS = [
    re.compile(p) for p in [
        r"^see\s",
        r"^e\.g\.[\s,]",
        r"^cf\.?\s",
        r"^as (?:described|discussed|shown|noted|reported|detailed) in\s",
        r"^\[?\d+[\],\-]",  # bare citation marker as the whole sentence
    ]
]

# Pattern matching citation-only content: author names + years + separators
# e.g. "Assran et al. (2023); Baevski et al. (2022); Bardes et al. (2024)"
_CITATION_ONLY_RE = re.compile(
    r"^[\s;,\.and&]+$"  # after stripping author-year groups, only separators remain
)
_AUTHOR_YEAR_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+|[A-Z]\.?))*"
    r"\s*\(?\d{4}[a-z]?\)?"
)


def is_substantive_citation(citation: Citation) -> bool:
    """Filter out trivial citations that make no verifiable claim.

    Skips:
    - Very short sentences (< 5 words)
    - Trivial phrases: 'see [5]', 'as described in [5]'
    - Citation-only sentences: 'Assran et al. (2023); Baevski et al. (2022)'
      (all content is author names + years with no actual claim)
    """
    sentence = citation.citing_sentence.strip()
    lower = sentence.lower()
    if len(lower.split()) < 5:
        return False

    for pattern in _TRIVIAL_PATTERNS:
        if pattern.match(lower):
            return False

    # Check if the sentence is nothing but citation references (author-year groups).
    # Strip all author-year patterns and see if anything meaningful remains.
    stripped = _AUTHOR_YEAR_RE.sub("", sentence)
    # Also strip the citation marker itself
    if citation.marker:
        stripped = stripped.replace(citation.marker, "")
    # Remove leftover separators, brackets, whitespace
    stripped = re.sub(r"[\s;,\.\(\)\[\]&]+", " ", stripped).strip()
    if len(stripped.split()) < 3:
        return False

    return True
