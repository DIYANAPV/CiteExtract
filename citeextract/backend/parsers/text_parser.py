
import re
from pathlib import Path

from citeextract.citation.context_extractor import extract_context
from citeextract.citation.detector import CitationDetector
from citeextract.models.citation import Citation
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.reference import Reference
from citeextract.parsers.base import BaseParser
from citeextract.parsers.marker_rule_check import annotate_citation_confidence


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
    sent_boundary = re.search(r'(?<![A-Z])\.\s+([A-Z])', text)
    if sent_boundary and sent_boundary.start() > 5:
        title = text[:sent_boundary.start()].strip().rstrip('.')
        if len(title) > 3:
            return title

    best_end = len(text)
    for pattern in _TITLE_END_PATTERNS:
        match = pattern.search(text)
        if match and match.start() < best_end:
            best_end = match.start()

    if best_end < len(text):
        title = text[:best_end].strip().rstrip('.')
        if len(title) > 3:
            return title

    title = text.strip().rstrip('.')
    return title if len(title) > 3 else None


def _parse_author_string(text: str) -> list[str]:
    parts = re.split(r',?\s+and\s+|\s*&\s*', text)
    authors: list[str] = []

    for part in parts:
        part = part.strip().rstrip(',')
        if not part or len(part) < 2:
            continue

        sub_authors = re.findall(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]\.(?:\s*[A-Z]\.)*)', part
        )
        if len(sub_authors) >= 2:
            authors.extend(a.strip().rstrip(',') for a in sub_authors)
        else:
            authors.append(part)

    return [a for a in authors if len(a) > 2]


_REF_SECTION_PATTERN = re.compile(
    r'^(?:references|bibliography|works\s+cited|literature\s+cited)\s*$',
    re.IGNORECASE | re.MULTILINE,
)


class TextParser(BaseParser):

    def can_parse(self, file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix in (".txt", "")

    def parse(self, file_path: str) -> ParsedPaper:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text)

    def parse_text(self, text: str) -> ParsedPaper:
        warnings: list[str] = []

        body_text, ref_section = self._split_sections(text)

        references: list[Reference] = []
        if ref_section:
            references = self._parse_reference_section(ref_section)
        else:
            warnings.append(
                "No reference section detected. "
                "Looked for 'References', 'Bibliography', 'Works Cited' heading."
            )

        has_body = bool(body_text.strip())

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

        if citations and references:
            refs_by_id = {r.ref_id: r for r in references}
            citations = annotate_citation_confidence(citations, refs_by_id)

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
        match = _REF_SECTION_PATTERN.search(text)
        if match:
            body = text[: match.start()].strip()
            ref_section = text[match.end() :].strip()
            return body, ref_section
        return text, ""

    def _parse_reference_section(self, section: str) -> list[Reference]:
        entries = re.split(r'\n\s*(?=\[\d+\]|\d+\.?\s+[A-Z])', section)
        if len(entries) <= 1:
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
        clean = re.sub(r'^\[\d+\]\s*', '', text)
        clean = re.sub(r'^\d+\.\s*', '', clean)

        if len(clean) < 10:
            return None

        year: int | None = None
        year_match = re.search(r'\b((?:19|20)\d{2})\b', clean)
        if year_match:
            year = int(year_match.group(1))

        doi: str | None = None
        doi_match = re.search(r'(10\.\d{4,}/[^\s,)\]}>]+)', clean)
        if doi_match:
            doi = doi_match.group(1).rstrip('.')

        title: str | None = None
        quoted = re.search(r'["\u201c](.+?)["\u201d]', clean)
        if quoted:
            title = quoted.group(1)
        else:
            if year_match:
                rest = clean[year_match.end():].strip().lstrip('). ')
                if rest:
                    title = _extract_title_from_rest(rest)

            if not title:
                segments = [s.strip() for s in clean.split('.') if len(s.strip()) > 15]
                if segments:
                    title = segments[1] if len(segments) > 1 else segments[0]

        authors: list[str] = []
        if year_match:
            author_text = clean[: year_match.start()].strip().rstrip('(,.')
            if author_text and len(author_text) > 3:
                authors = _parse_author_string(author_text)

        url: str | None = None
        url_match = re.search(r'https?://\S+', clean)
        if url_match:
            url = url_match.group(0).rstrip('.,;)')

        from citeextract.citation.format_detector import detect_citation_format

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
