"""Format router — detects input format and dispatches to the right parser."""

import logging
import re
from pathlib import Path

from src.models.parsed_paper import ParsedPaper

log = logging.getLogger(__name__)

# Input limits to prevent runaway API calls and OOM
MAX_FILE_SIZE_MB = 50
MAX_REFERENCES = 500


class UnsupportedFormatError(Exception):
    """Raised for file types we can't parse."""


class InputTooLargeError(Exception):
    """Raised when input exceeds size or reference count limits."""


def parse_file(file_path: str) -> ParsedPaper:
    """Detect format by extension and route to appropriate parser.

    Supported: .pdf, .tex, .bib, .txt
    PDF parsing requires GROBID (Docker container).

    Args:
        file_path: Path to the input file.

    Returns:
        ParsedPaper with normalized references and citations.

    Raises:
        UnsupportedFormatError: If file extension is not supported.
        InputTooLargeError: If file exceeds size or reference count limits.
        FileNotFoundError: If file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Validate file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise InputTooLargeError(
            f"File is {size_mb:.1f}MB, exceeds {MAX_FILE_SIZE_MB}MB limit."
        )

    ext = path.suffix.lower()

    if ext == ".bib":
        from src.parsers.bibtex_parser import BibtexParser
        result = BibtexParser().parse(file_path)
    elif ext == ".tex":
        from src.parsers.latex_parser import LatexParser
        result = LatexParser().parse(file_path)
    elif ext == ".pdf":
        result = _parse_pdf(file_path)
    elif ext in (".txt", ""):
        from src.parsers.text_parser import TextParser
        result = TextParser().parse(file_path)
    else:
        raise UnsupportedFormatError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: .pdf, .tex, .bib, .txt"
        )

    result = _deduplicate_references(result)

    # Validate reference count
    if len(result.references) > MAX_REFERENCES:
        raise InputTooLargeError(
            f"Paper has {len(result.references)} references, "
            f"exceeds {MAX_REFERENCES} limit."
        )

    return result


def _deduplicate_references(result: ParsedPaper) -> ParsedPaper:
    """Merge references with identical normalized titles.

    Keeps the reference with more metadata fields populated.
    Remaps citation ref_ids to the kept reference.
    """
    if len(result.references) <= 1:
        return result

    def _norm(title: str) -> str:
        t = title.lower().strip()
        t = re.sub(r'[^\w\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    # Group by normalized title
    groups: dict[str, list] = {}
    for ref in result.references:
        if not ref.title:
            groups[f"__notitle_{ref.ref_id}"] = [ref]
            continue
        key = _norm(ref.title)
        if key not in groups:
            groups[key] = []
        groups[key].append(ref)

    # Check if any dedup needed
    has_dupes = any(len(refs) > 1 for refs in groups.values())
    if not has_dupes:
        return result

    # Pick best reference per group, build remap
    kept_refs = []
    remap: dict[str, str] = {}  # old_ref_id -> new_ref_id

    def _metadata_score(ref) -> int:
        """Count how many metadata fields are populated."""
        score = 0
        if ref.title: score += 1
        if ref.authors: score += 1
        if ref.year: score += 1
        if ref.venue: score += 1
        if ref.doi: score += 2  # DOI is extra valuable
        if ref.arxiv_id: score += 1
        return score

    for refs in groups.values():
        if len(refs) == 1:
            kept_refs.append(refs[0])
            continue
        # Sort by metadata richness, keep the best
        refs.sort(key=_metadata_score, reverse=True)
        best = refs[0]
        kept_refs.append(best)
        for dup in refs[1:]:
            remap[dup.ref_id] = best.ref_id

    # Remap citations
    new_citations = []
    for cit in result.citations:
        if cit.ref_id in remap:
            cit = cit.model_copy(update={"ref_id": remap[cit.ref_id]})
        new_citations.append(cit)

    # Build warning
    warnings = list(result.warnings)
    if remap:
        warnings.append(
            f"Deduplicated {len(remap)} reference(s) with identical titles."
        )

    return ParsedPaper(
        references=kept_refs,
        citations=new_citations,
        has_body_text=result.has_body_text,
        body_text=result.body_text,
        input_format=result.input_format,
        metadata=result.metadata,
        warnings=warnings,
    )


def _parse_pdf(file_path: str) -> ParsedPaper:
    """PDF parsing via GROBID.

    GROBID must be running as a Docker container. If it's not available,
    parsing fails with a clear error message.
    """
    from src.parsers.grobid_parser import GrobidParser, GrobidError

    try:
        return GrobidParser().parse(file_path)
    except (GrobidError, ConnectionError, OSError) as e:
        log.error(
            "GROBID is not available — cannot parse PDF. "
            "Start GROBID with: docker run -d --name grobid "
            "-p 8070:8070 grobid/grobid:0.8.2-crf"
        )
        raise GrobidError(
            f"PDF parsing requires GROBID but it is not available: {e}. "
            f"Start GROBID with: docker run -d --name grobid "
            f"-p 8070:8070 grobid/grobid:0.8.2-crf"
        ) from e
