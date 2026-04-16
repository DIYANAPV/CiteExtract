"""Citation format detection — identifies bibliography style from raw reference text.

Detects: APA, Vancouver, IEEE, Chicago, Harvard, MLA, or None (unknown).
Each format has different rules for author naming, truncation thresholds,
and structural patterns that affect how we compare metadata.
"""

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Format rules — truncation thresholds and expected name styles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatRules:
    """Rules for a specific citation format."""

    name: str
    et_al_after: Optional[int]  # authors truncated after this count (None = no rule)
    name_style: str             # expected author name format description


FORMAT_RULES: dict[str, FormatRules] = {
    "apa": FormatRules(
        name="APA",
        et_al_after=20,
        name_style="Surname, A. B.",
    ),
    "vancouver": FormatRules(
        name="Vancouver",
        et_al_after=6,
        name_style="Surname AB",
    ),
    "ieee": FormatRules(
        name="IEEE",
        et_al_after=6,
        name_style="A. B. Surname",
    ),
    "chicago": FormatRules(
        name="Chicago",
        et_al_after=3,
        name_style="Surname, Firstname",
    ),
    "harvard": FormatRules(
        name="Harvard",
        et_al_after=3,
        name_style="Surname, A.B.",
    ),
    "mla": FormatRules(
        name="MLA",
        et_al_after=1,
        name_style="Surname, Firstname",
    ),
    "bibtex": FormatRules(
        name="BibTeX",
        et_al_after=None,
        name_style="varies",
    ),
}


def get_format_rules(fmt: Optional[str]) -> Optional[FormatRules]:
    """Get the rules for a detected format, or None if unknown."""
    if fmt is None:
        return None
    return FORMAT_RULES.get(fmt)


# ---------------------------------------------------------------------------
# Format detection — pattern-based scoring
# ---------------------------------------------------------------------------

# Minimum score to accept a format detection (out of max possible per format)
_MIN_CONFIDENCE = 2


def detect_citation_format(raw_text: str, source_format: str = "") -> Optional[str]:
    """Detect the bibliography citation format from raw reference text.

    Args:
        raw_text: The original reference string as it appears in the document.
        source_format: The parser that produced this reference (grobid, bibtex, etc.).

    Returns:
        One of "apa", "vancouver", "ieee", "chicago", "harvard", "mla", "bibtex",
        or None if format cannot be determined.
    """
    if source_format == "bibtex":
        return "bibtex"

    if not raw_text or len(raw_text.strip()) < 15:
        return None

    text = raw_text.strip()

    scores: dict[str, int] = {
        "apa": _score_apa(text),
        "vancouver": _score_vancouver(text),
        "ieee": _score_ieee(text),
        "chicago": _score_chicago(text),
        "harvard": _score_harvard(text),
        "mla": _score_mla(text),
    }

    best_format = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_format]

    if best_score < _MIN_CONFIDENCE:
        return None

    # Tie-breaking: if top two are equal, return None (ambiguous)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        return None

    return best_format


def _score_apa(text: str) -> int:
    """Score how likely the text is APA format.

    APA pattern: Author, A. B., & Author, C. D. (Year). Title. Journal, Vol(Issue), pages.
    """
    score = 0

    # Year in parentheses after authors: (2020)
    if re.search(r'\.\s*\(\d{4}\)', text):
        score += 2

    # Ampersand before last author: & Author
    if re.search(r'&\s+[A-Z]', text):
        score += 2

    # Author initials with periods: A. B.
    if re.search(r'[A-Z]\.\s*[A-Z]\.', text):
        score += 1

    # Surname, Initial pattern: Smith, J.
    if re.search(r'[A-Z][a-z]+,\s+[A-Z]\.', text):
        score += 1

    # Italic-style journal (often followed by comma and volume): Journal, 12(3)
    if re.search(r',\s*\d+\(\d+\)', text):
        score += 1

    # DOI URL at end
    if re.search(r'https?://doi\.org/', text):
        score += 1

    return score


def _score_vancouver(text: str) -> int:
    """Score how likely the text is Vancouver format.

    Vancouver pattern: Author AB, Author CD. Title. Journal. Year;Vol(Issue):pages.
    """
    score = 0

    # Author initials WITHOUT periods right after surname: Huang H, LeCun Y
    if re.search(r'[A-Z][a-z]+\s+[A-Z]{1,3}[,.]', text):
        score += 3

    # Semicolon before volume number: 2020;45(2)
    if re.search(r'\d{4};\d+', text):
        score += 2

    # Colon before page numbers: :123-130
    if re.search(r':\d+-\d+', text):
        score += 1

    # No ampersand (Vancouver uses commas throughout)
    if '&' not in text:
        score += 1

    # "et al." (Vancouver uses "et al." not "et al.,")
    if 'et al.' in text and 'et al.,' not in text:
        score += 1

    return score


def _score_ieee(text: str) -> int:
    """Score how likely the text is IEEE format.

    IEEE pattern: A. B. Author, C. D. Author, "Title," in Proc. Conf, Year, pp. X-Y.
    """
    score = 0

    # Title in double quotes: "Title"
    if re.search(r'"[^"]{10,}"', text):
        score += 2

    # Initials before surname: A. B. Author or A. Author
    if re.search(r'^[A-Z]\.\s*(?:[A-Z]\.\s*)?[A-Z][a-z]', text):
        score += 3

    # "in Proc." or "in IEEE" or "in Int."
    if re.search(r'\bin\s+(Proc\.|IEEE|Int\.|ACM)', text):
        score += 2

    # "pp." for page numbers
    if re.search(r'\bpp\.\s*\d+', text):
        score += 1

    # "vol." for volume
    if re.search(r'\bvol\.\s*\d+', text, re.IGNORECASE):
        score += 1

    return score


def _score_chicago(text: str) -> int:
    """Score how likely the text is Chicago format.

    Chicago pattern: Author, Firstname. "Title." Journal Vol, no. Issue (Year): pages.
    """
    score = 0

    # Title in double quotes followed by period: "Title."
    if re.search(r'"[^"]{10,}"\s*\.', text):
        score += 2

    # "no." for issue number
    if re.search(r'\bno\.\s*\d+', text):
        score += 2

    # Full first name after surname: Author, Firstname
    # (at least 3 chars after comma, no period immediately after)
    if re.search(r'^[A-Z][a-z]+,\s+[A-Z][a-z]{2,}', text):
        score += 2

    # Year in parentheses with colon after: (2020):
    if re.search(r'\(\d{4}\)\s*:', text):
        score += 2

    return score


def _score_harvard(text: str) -> int:
    """Score how likely the text is Harvard format.

    Harvard pattern: Author, A.B. Year, 'Title', Journal, vol. X, no. Y, pp. Z.
    """
    score = 0

    # Year immediately after author (no parentheses): Author, A.B. 2020,
    if re.search(r'[A-Z]\.\s*\d{4}[,.]', text):
        score += 2

    # Title in single quotes: 'Title'
    if re.search(r"'[^']{10,}'", text):
        score += 2

    # "vol." and "no." and "pp." all present
    lower = text.lower()
    vol_no_pp = sum(1 for kw in ['vol.', 'no.', 'pp.'] if kw in lower)
    if vol_no_pp >= 2:
        score += 2

    # Comma after year (not parenthesized): 2020,
    if re.search(r'\d{4},\s', text):
        score += 1

    return score


def _score_mla(text: str) -> int:
    """Score how likely the text is MLA format.

    MLA pattern: Author. "Title." Journal, vol. X, no. Y, Year, pp. Z.
    """
    score = 0

    # Title in double quotes: "Title."
    if re.search(r'"[^"]{10,}"', text):
        score += 1

    # Full first name after surname with period: Surname, Firstname.
    if re.search(r'^[A-Z][a-z]+,\s+[A-Z][a-z]{2,}\.', text):
        score += 2

    # Lowercase "vol." and "no."
    if re.search(r'\bvol\.\s*\d+', text) and 'Vol.' not in text:
        score += 1

    # Year near end, not in parentheses
    if re.search(r',\s*\d{4}[,.]', text):
        score += 1

    # No ampersand (MLA doesn't use &)
    if '&' not in text:
        score += 1

    # "pp." for pages (MLA 8th doesn't always use pp., but common in earlier)
    if re.search(r'\bpp\.\s*\d+', text):
        score += 1

    return score
