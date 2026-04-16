"""Plain text parser — best-effort extraction from unstructured text.

Detects citation markers in body text and tries to identify a reference
section. Quality depends entirely on how well-formatted the input is.
"""

import re
from pathlib import Path

from src.citation.context_extractor import extract_context
from src.citation.detector import CitationDetector
from src.models.citation import Citation
from src.models.parsed_paper import ParsedPaper
from src.models.reference import Reference
from src.parsers.base import BaseParser


# Patterns marking the end of a title in a reference string
_TITLE_END_PATTERNS = [
    re.compile(r'\.\s+arXiv\s+preprint', re.IGNORECASE),
    re.compile(r'\.\s+In\s+(?:Proceedings|Proc\.)', re.IGNORECASE),
    re.compile(r'\.\s+(?:IEEE|ACM|AAAI|NeurIPS|ICML|ICLR|CVPR|ECCV|ICCV|EMNLP|ACL|NAACL)', re.IGNORECASE),
    re.compile(r'\.\s+(?:Journal|Transactions|Conference)', re.IGNORECASE),
    re.compile(r',\s+pages?\s+', re.IGNORECASE),
    re.compile(r',\s+pp\.\s+', re.IGNORECASE),
    re.compile(r',\s+20\d{2}\b'),
    re.compile(r',\s+19\d{2}\b'),
    re.compile(r'\.\s+URL\s+', re.IGNORECASE),
    re.compile(r'\.\s+doi:', re.IGNORECASE),
]


def _extract_title_from_rest(text: str) -> str | None:
    """Extract the title from text that follows the year/author portion.

    In standard academic reference formats, the title is the first sentence
    after the year: "Authors (Year). Title. Venue, Volume, Pages."

    Strategy:
    1. Look for a sentence boundary (". " followed by uppercase letter) — the
       title ends at the first such boundary. This handles the vast majority of
       reference formats without needing an exhaustive venue list.
    2. Fall back to known venue/metadata patterns for edge cases.
    3. Last resort: take everything.
    """
    # Strategy 1: First sentence boundary — ". " followed by uppercase
    # Skip abbreviation-like periods (single letter before period, e.g. "A. N.")
    sent_boundary = re.search(r'(?<![A-Z])\.\s+([A-Z])', text)
    if sent_boundary and sent_boundary.start() > 5:
        title = text[:sent_boundary.start()].strip().rstrip('.')
        if len(title) > 3:
            return title

    # Strategy 2: Known venue/metadata patterns
    best_end = len(text)
    for pattern in _TITLE_END_PATTERNS:
        match = pattern.search(text)
        if match and match.start() < best_end:
            best_end = match.start()

    if best_end < len(text):
        title = text[:best_end].strip().rstrip('.')
        if len(title) > 3:
            return title

    # Strategy 3: Take everything (strip trailing period)
    title = text.strip().rstrip('.')
    return title if len(title) > 3 else None


def _parse_author_string(text: str) -> list[str]:
    """Parse an author string in common formats.

    Handles:
    - "Last, F., Last2, F., and Last3, F."  (comma-separated with initials)
    - "Last, First and Last2, First2"        (BibTeX-like)
    - "F. Last, F. Last2, and F. Last3"      (initial-first)
    """
    # Split on " and " or " & " first
    parts = re.split(r',?\s+and\s+|\s*&\s*', text)
    authors: list[str] = []

    for part in parts:
        part = part.strip().rstrip(',')
        if not part or len(part) < 2:
            continue

        # Check if this part contains multiple "Last, I." entries
        # Pattern: "Surname, I." repeated with commas
        # E.g. "Vaswani, A., Shazeer, N., Parmar, N."
        sub_authors = re.findall(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]\.(?:\s*[A-Z]\.)*)', part
        )
        if len(sub_authors) >= 2:
            authors.extend(a.strip().rstrip(',') for a in sub_authors)
        else:
            authors.append(part)

    return [a for a in authors if len(a) > 2]


# Headings that mark the start of a reference section
_REF_SECTION_PATTERN = re.compile(
    r'^(?:references|bibliography|works\s+cited|literature\s+cited)\s*$',
    re.IGNORECASE | re.MULTILINE,
)


class TextParser(BaseParser):
    """Best-effort parser for plain text input."""

    def can_parse(self, file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix in (".txt", "")

    def parse(self, file_path: str) -> ParsedPaper:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text)

    def parse_text(self, text: str) -> ParsedPaper:
        """Parse raw text string (used by router for piped/pasted text too)."""
        warnings: list[str] = []

        # --- Split body vs reference section ---
        body_text, ref_section = self._split_sections(text)

        # --- Parse references from the reference section ---
        references: list[Reference] = []
        if ref_section:
            references = self._parse_reference_section(ref_section)
        else:
            warnings.append(
                "No reference section detected. "
                "Looked for 'References', 'Bibliography', 'Works Cited' heading."
            )

        has_body = bool(body_text.strip())

        # --- Detect citations in body text ---
        citations: list[Citation] = []
        if has_body:
            detector = CitationDetector()
            detected = detector.detect_all(body_text)
            for det in detected:
                ctx = extract_context(body_text, det.position)
                for key in det.keys:
                    citations.append(Citation(
                        ref_id=key,
                        citing_sentence=ctx["citing_sentence"],
                        context_before=ctx["context_before"],
                        context_after=ctx["context_after"],
                        marker=det.marker,
                        position=det.position,
                    ))

        if not references and not citations:
            warnings.append("No references or citations found. Check text formatting.")

        warnings.append(
            "Plain text input: extraction quality depends on formatting."
        )

        return ParsedPaper(
            references=references,
            citations=citations,
            has_body_text=has_body,
            body_text=body_text if has_body else "",
            input_format="text",
            metadata={},
            warnings=warnings,
        )

    def _split_sections(self, text: str) -> tuple[str, str]:
        """Split text into body and reference section."""
        match = _REF_SECTION_PATTERN.search(text)
        if match:
            body = text[: match.start()].strip()
            ref_section = text[match.end() :].strip()
            return body, ref_section
        return text, ""

    def _parse_reference_section(self, section: str) -> list[Reference]:
        """Parse individual references from a reference section.

        Splits on blank lines or numbered patterns like '[1]' or '1.'
        """
        # Try splitting by numbered patterns first
        entries = re.split(r'\n\s*(?=\[\d+\]|\d+\.?\s+[A-Z])', section)
        if len(entries) <= 1:
            # Fallback: split by blank lines
            entries = re.split(r'\n\s*\n', section)

        references: list[Reference] = []
        for i, entry in enumerate(entries, start=1):
            entry = entry.strip()
            if len(entry) < 10:
                continue

            ref = self._parse_single_reference(entry, str(i))
            if ref is not None:
                references.append(ref)

        return references

    def _parse_single_reference(self, text: str, ref_id: str) -> Reference | None:
        """Best-effort extraction of fields from a single reference string."""
        # Strip leading number/bracket: "[1] " or "1. "
        clean = re.sub(r'^\[\d+\]\s*', '', text)
        clean = re.sub(r'^\d+\.\s*', '', clean)

        if len(clean) < 10:
            return None

        # Extract year (4-digit number)
        year: int | None = None
        year_match = re.search(r'\b((?:19|20)\d{2})\b', clean)
        if year_match:
            year = int(year_match.group(1))

        # Extract DOI
        doi: str | None = None
        doi_match = re.search(r'(10\.\d{4,}/[^\s,)\]}>]+)', clean)
        if doi_match:
            doi = doi_match.group(1).rstrip('.')

        # Title heuristic: text in quotes, or text after year before venue markers
        title: str | None = None
        quoted = re.search(r'["\u201c](.+?)["\u201d]', clean)
        if quoted:
            title = quoted.group(1)
        else:
            # Try: text after authors (after year/period) before venue markers
            if year_match:
                rest = clean[year_match.end():].strip().lstrip('). ')
                if rest:
                    title = _extract_title_from_rest(rest)

            if not title:
                # Fallback: take the longest period-delimited segment
                segments = [s.strip() for s in clean.split('.') if len(s.strip()) > 15]
                if segments:
                    title = segments[1] if len(segments) > 1 else segments[0]

        # Authors heuristic: text before the year
        authors: list[str] = []
        if year_match:
            author_text = clean[: year_match.start()].strip().rstrip('(,.')
            if author_text and len(author_text) > 3:
                authors = _parse_author_string(author_text)

        # Extract URL
        url: str | None = None
        url_match = re.search(r'https?://\S+', clean)
        if url_match:
            url = url_match.group(0).rstrip('.,;)')

        from src.citation.format_detector import detect_citation_format

        return Reference(
            ref_id=ref_id,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            url=url,
            raw_text=text,
            source_format="text",
            citation_format=detect_citation_format(text, "text"),
        )
