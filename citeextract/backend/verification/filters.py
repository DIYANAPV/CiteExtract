
import re

from citeextract.models.citation import Citation


_TRIVIAL_PATTERNS = [
    re.compile(p) for p in [
        r"^see\s",
        r"^e\.g\.[\s,]",
        r"^cf\.?\s",
        r"^as (?:described|discussed|shown|noted|reported|detailed) in\s",
        r"^\[?\d+[\],\-]",
    ]
]

_CITATION_ONLY_RE = re.compile(
    r"^[\s;,\.and&]+$"
)
_AUTHOR_YEAR_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+|[A-Z]\.?))*"
    r"\s*\(?\d{4}[a-z]?\)?"
)


def is_substantive_citation(citation: Citation) -> bool:
    sentence = citation.citing_sentence.strip()
    lower = sentence.lower()
    if len(lower.split()) < 5:
        return False

    for pattern in _TRIVIAL_PATTERNS:
        if pattern.match(lower):
            return False

    stripped = _AUTHOR_YEAR_RE.sub("", sentence)
    if citation.marker:
        stripped = stripped.replace(citation.marker, "")
    stripped = re.sub(r"[\s;,\.\(\)\[\]&]+", " ", stripped).strip()
    if len(stripped.split()) < 3:
        return False

    return True
