
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FormatRules:

    name: str
    et_al_after: Optional[int]
    name_style: str


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
    if fmt is None:
        return None
    return FORMAT_RULES.get(fmt)


_MIN_CONFIDENCE = 2


def detect_citation_format(raw_text: str, source_format: str = "") -> Optional[str]:
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

    best_format = max(scores, key=scores.get)
    best_score = scores[best_format]

    if best_score < _MIN_CONFIDENCE:
        return None

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        return None

    return best_format


def _score_apa(text: str) -> int:
    score = 0

    if re.search(r'\.\s*\(\d{4}\)', text):
        score += 2

    if re.search(r'&\s+[A-Z]', text):
        score += 2

    if re.search(r'[A-Z]\.\s*[A-Z]\.', text):
        score += 1

    if re.search(r'[A-Z][a-z]+,\s+[A-Z]\.', text):
        score += 1

    if re.search(r',\s*\d+\(\d+\)', text):
        score += 1

    if re.search(r'https?://doi\.org/', text):
        score += 1

    return score


def _score_vancouver(text: str) -> int:
    score = 0

    if re.search(r'[A-Z][a-z]+\s+[A-Z]{1,3}[,.]', text):
        score += 3

    if re.search(r'\d{4};\d+', text):
        score += 2

    if re.search(r':\d+-\d+', text):
        score += 1

    if '&' not in text:
        score += 1

    if 'et al.' in text and 'et al.,' not in text:
        score += 1

    return score


def _score_ieee(text: str) -> int:
    score = 0

    if re.search(r'"[^"]{10,}"', text):
        score += 2

    if re.search(r'^[A-Z]\.\s*(?:[A-Z]\.\s*)?[A-Z][a-z]', text):
        score += 3

    if re.search(r'\bin\s+(Proc\.|IEEE|Int\.|ACM)', text):
        score += 2

    if re.search(r'\bpp\.\s*\d+', text):
        score += 1

    if re.search(r'\bvol\.\s*\d+', text, re.IGNORECASE):
        score += 1

    return score


def _score_chicago(text: str) -> int:
    score = 0

    if re.search(r'"[^"]{10,}"\s*\.', text):
        score += 2

    if re.search(r'\bno\.\s*\d+', text):
        score += 2

    if re.search(r'^[A-Z][a-z]+,\s+[A-Z][a-z]{2,}', text):
        score += 2

    if re.search(r'\(\d{4}\)\s*:', text):
        score += 2

    return score


def _score_harvard(text: str) -> int:
    score = 0

    if re.search(r'[A-Z]\.\s*\d{4}[,.]', text):
        score += 2

    if re.search(r"'[^']{10,}'", text):
        score += 2

    lower = text.lower()
    vol_no_pp = sum(1 for kw in ['vol.', 'no.', 'pp.'] if kw in lower)
    if vol_no_pp >= 2:
        score += 2

    if re.search(r'\d{4},\s', text):
        score += 1

    return score


def _score_mla(text: str) -> int:
    score = 0

    if re.search(r'"[^"]{10,}"', text):
        score += 1

    if re.search(r'^[A-Z][a-z]+,\s+[A-Z][a-z]{2,}\.', text):
        score += 2

    if re.search(r'\bvol\.\s*\d+', text) and 'Vol.' not in text:
        score += 1

    if re.search(r',\s*\d{4}[,.]', text):
        score += 1

    if '&' not in text:
        score += 1

    if re.search(r'\bpp\.\s*\d+', text):
        score += 1

    return score
