
import re
from pathlib import Path

import bibtexparser

from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.reference import Reference
from citeextract.parsers.base import BaseParser


_LATEX_UNICODE_MAP = {
    r'\"a': 'ä', r'\"o': 'ö', r'\"u': 'ü',
    r'\"A': 'Ä', r'\"O': 'Ö', r'\"U': 'Ü',
    r"\'a": 'á', r"\'e": 'é', r"\'i": 'í', r"\'o": 'ó', r"\'u": 'ú',
    r"\'A": 'Á', r"\'E": 'É', r"\'I": 'Í', r"\'O": 'Ó', r"\'U": 'Ú',
    r'\`a': 'à', r'\`e': 'è', r'\`i': 'ì', r'\`o': 'ò', r'\`u': 'ù',
    r'\^a': 'â', r'\^e': 'ê', r'\^i': 'î', r'\^o': 'ô', r'\^u': 'û',
    r'\~n': 'ñ', r'\~a': 'ã', r'\~o': 'õ',
    r'\c{c}': 'ç', r'\c{C}': 'Ç',
    r'\v{c}': 'č', r'\v{s}': 'š', r'\v{z}': 'ž',
    r'\v{C}': 'Č', r'\v{S}': 'Š', r'\v{Z}': 'Ž',
    r'\L': 'Ł', r'\l': 'ł',
    r'\o': 'ø', r'\O': 'Ø',
    r'\aa': 'å', r'\AA': 'Å',
    r'\ae': 'æ', r'\AE': 'Æ',
    r'\ss': 'ß',
}

_LATEX_ACCENT_BRACED = re.compile(r"""\\(["'`^~])(\{[a-zA-Z]\})""")
_LATEX_ACCENT_BARE = re.compile(r"""\\(["'`^~])([a-zA-Z])""")


def _clean_latex_str(text: str) -> str:
    def _debrace(m: re.Match) -> str:
        letter = m.group(2).strip('{}')
        key = f'\\{m.group(1)}{letter}'
        return _LATEX_UNICODE_MAP.get(key, letter)

    text = _LATEX_ACCENT_BRACED.sub(_debrace, text)

    for latex, uni in sorted(_LATEX_UNICODE_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(latex, uni)

    text = text.replace('{', '').replace('}', '')
    return text


class BibtexParser(BaseParser):

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".bib"

    def parse(self, file_path: str) -> ParsedPaper:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bib_db = bibtexparser.loads(text, parser=parser)

        references: list[Reference] = []
        warnings: list[str] = []

        for entry in bib_db.entries:
            ref = self._entry_to_reference(entry)
            if ref is not None:
                references.append(ref)

        warnings.append(
            "BibTeX input: semantic verification will be skipped (no citing context)"
        )

        return ParsedPaper(
            references=references,
            citations=[],
            has_body_text=False,
            input_format="bibtex",
            metadata={},
            warnings=warnings,
        )

    def _entry_to_reference(self, entry: dict) -> Reference | None:
        title = _clean_latex_str(entry.get("title", "").strip(" {}"))
        if not title:
            return None

        authors = self._parse_authors(entry.get("author", ""))
        year = self._parse_year(entry.get("year", ""))
        venue = (
            entry.get("journal")
            or entry.get("booktitle")
            or entry.get("publisher")
            or ""
        )
        venue = _clean_latex_str(venue.strip(" {}"))
        doi = entry.get("doi", "").strip()
        ref_id = entry.get("ID", "unknown")

        url = entry.get("url", "").strip()
        if not url:
            howpub = entry.get("howpublished", "")
            url_match = re.search(r'https?://\S+', howpub)
            if url_match:
                url = url_match.group(0).rstrip('.,;)}')
        if not url:
            note = entry.get("note", "")
            url_match = re.search(r'https?://\S+', note)
            if url_match:
                url = url_match.group(0).rstrip('.,;)}')

        raw_parts = []
        if authors:
            raw_parts.append(", ".join(authors[:3]))
            if len(authors) > 3:
                raw_parts[-1] += " et al."
        if year:
            raw_parts.append(f"({year})")
        raw_parts.append(title)
        if venue:
            raw_parts.append(venue)

        return Reference(
            ref_id=ref_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi if doi else None,
            url=url if url else None,
            raw_text=". ".join(raw_parts),
            source_format="bibtex",
            citation_format="bibtex",
        )

    @staticmethod
    def _parse_authors(author_str: str) -> list[str]:
        if not author_str:
            return []
        author_str = author_str.strip(" {}")
        parts = re.split(r'\s+and\s+', author_str, flags=re.IGNORECASE)
        authors: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ',' in part:
                pieces = part.split(',', 1)
                name = f"{pieces[1].strip()} {pieces[0].strip()}"
            else:
                name = part
            name = _clean_latex_str(name.strip())
            if name:
                authors.append(name)
        return authors

    @staticmethod
    def _parse_year(year_str: str) -> int | None:
        year_str = year_str.strip(" {}")
        match = re.search(r'(\d{4})', year_str)
        if match:
            return int(match.group(1))
        return None
