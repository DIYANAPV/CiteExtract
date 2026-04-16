"""GROBID PDF parser — primary parser for PDF input.

Sends PDFs to a local GROBID service for TEI XML extraction, then parses
the structured XML to produce references, citations, and full text.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import config
from src.citation.context_extractor import extract_context
from src.citation.detector import CitationDetector, normalize_author_name
from src.models.citation import Citation
from src.models.parsed_paper import ParsedPaper
from src.models.reference import Reference
from src.parsers.base import BaseParser

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidError(Exception):
    """Raised when GROBID service is unavailable or fails."""


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def _is_self_reference(ref_title: str, paper_title: str) -> bool:
    """Check if a reference is a self-citation of the paper being analyzed."""
    from difflib import SequenceMatcher  # stdlib, no cost to lazy import
    ref_norm = _normalize_text(ref_title)
    paper_norm = _normalize_text(paper_title)
    if not ref_norm or not paper_norm:
        return False
    if paper_norm in ref_norm:
        return True
    return SequenceMatcher(None, ref_norm, paper_norm).ratio() > config.thresholds()["title_match"]


def _is_valid_reference(ref_data: dict) -> bool:
    """Check if extracted reference has enough fields to be useful."""
    if ref_data.get("doi") or ref_data.get("arxiv_id"):
        return True
    has_year = bool(ref_data.get("year"))
    has_authors = len(ref_data.get("authors", [])) > 0
    has_title = ref_data.get("title") and len(ref_data["title"]) > 5
    if not has_year:
        year_in_raw = bool(re.search(r'\b(19|20)\d{2}\b', ref_data.get("raw_text", "")))
        if not year_in_raw:
            return False
    return has_authors or has_title


def _clean_grobid_reference(ref_data: dict) -> dict | None:
    """Conservative cleanup of GROBID-parsed reference data.

    Fixes known GROBID artifacts without being overly aggressive.
    Returns cleaned ref_data, or None to remove obvious garbage.
    """
    title = ref_data.get("title") or ""

    # --- Remove obvious non-references ---

    # Figure/table captions extracted as bibliography entries
    if re.match(r'^(Fig\.|Figure|Table|Plate)\s', title, re.IGNORECASE):
        return None

    # Very long "titles" with casual language = body text leakage
    # Requires BOTH conditions: long AND casual markers
    if len(title) > 150:
        casual = re.search(
            r'\b(scary|amazing|cool|weird|funny|basically|stuff|'
            r'things like|gonna|wanna|crazy|lol|btw)\b',
            title, re.IGNORECASE,
        )
        if casual:
            return None

    # --- Clean fields without removing ---

    # Venue == title: GROBID often copies the title into venue when it
    # can't find a journal/conference name. Null it out instead of
    # letting it cause a false venue MISMATCH downstream.
    venue = ref_data.get("venue") or ""
    if venue and title:
        from difflib import SequenceMatcher
        venue_norm = _normalize_text(venue)
        title_norm = _normalize_text(title)
        if venue_norm and title_norm:
            if (SequenceMatcher(None, venue_norm, title_norm).ratio() > 0.80
                    or venue_norm in title_norm
                    or title_norm in venue_norm):
                ref_data["venue"] = None

    return ref_data


class GrobidParser(BaseParser):
    """Parse PDFs via GROBID TEI XML for structured reference and text extraction."""

    def __init__(self):
        cfg = config.grobid()
        self.service_url = cfg["service_url"].rstrip("/")
        self.timeout = cfg["timeout"]
        self._health_check_timeout = cfg["health_check_timeout"]
        self._docker_image = cfg["docker_image"]
        self.session = requests.Session()
        concurrency = cfg["concurrency"]
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=concurrency, pool_maxsize=concurrency
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> ParsedPaper:
        """Parse a PDF through GROBID and produce a ParsedPaper.

        Two-level caching for deterministic results:
        1. ParsedPaper cache (data/cache/parsed_{hash}.json) — full result
           with references, citations, everything. If this exists, skip all
           parsing entirely. This guarantees identical ref_ids across runs.
        2. GROBID XML cache (data/cache/grobid_{hash}.xml) — raw TEI XML.
           Used when ParsedPaper cache doesn't exist (first run or cleared).
        """
        import hashlib
        import json as _json
        pdf_bytes = Path(file_path).read_bytes()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

        # Level 1: Full ParsedPaper cache — deterministic ref_ids
        parsed_cache = Path("data/cache") / f"parsed_{pdf_hash}.json"
        if parsed_cache.exists():
            try:
                cached_data = _json.loads(parsed_cache.read_text(encoding="utf-8"))
                return ParsedPaper(**cached_data)
            except Exception:
                pass  # corrupted cache, fall through to re-parse

        # Level 2: GROBID XML cache
        cache_path = Path("data/cache") / f"grobid_{pdf_hash}.xml"
        if cache_path.exists():
            xml_content = cache_path.read_text(encoding="utf-8")
        else:
            self._ensure_grobid()
            xml_content = self._process_pdf(file_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(xml_content, encoding="utf-8")

        root = ET.fromstring(xml_content)

        warnings: list[str] = []

        # Extract paper metadata
        metadata = self._extract_paper_metadata(root)
        paper_title = metadata.get("title", "")

        # Extract body text (GROBID is reliable for this)
        # Need a basic biblio_id_map for text extraction — build from XML positions
        listbibl = root.find(".//tei:listBibl", TEI_NS)
        basic_biblio_map: dict[str, str] = {}
        if listbibl is not None:
            for i, bib in enumerate(listbibl.findall(".//tei:biblStruct", TEI_NS), 1):
                bid = bib.get("{http://www.w3.org/XML/1998/namespace}id")
                if bid:
                    basic_biblio_map[bid] = str(i)

        full_text, _cited_ref_ids = self._extract_text_and_citations(root, basic_biblio_map)

        if not full_text:
            warnings.append("GROBID returned no body text for this PDF.")

        # --- Try LLM-based reference parsing (best quality) ---
        llm_result = self._try_llm_parsing(root, full_text, warnings)

        if llm_result is not None:
            references_dict, citations, extra_warnings = llm_result
            warnings.extend(extra_warnings)
        else:
            # --- Fallback: GROBID's own structured parsing ---
            references_dict, biblio_id_map = self._extract_bibliography(root, paper_title)

            # Use GROBID's target mapping for citations (reliable, 87% coverage)
            citations = self._build_citations_from_grobid_targets(
                root, full_text, references_dict, warnings,
                bid_override=biblio_id_map,
            )

        # Deduplicate references with identical titles
        references_dict, dedup_count = self._deduplicate_references(references_dict)
        if dedup_count > 0:
            warnings.append(f"Deduplicated {dedup_count} reference(s) with identical titles.")

        result = ParsedPaper(
            references=list(references_dict.values()),
            citations=citations,
            has_body_text=bool(full_text),
            body_text=full_text or "",
            input_format="pdf",
            metadata=metadata,
            warnings=warnings,
        )

        # Cache the full ParsedPaper for deterministic ref_ids on future runs
        try:
            parsed_cache.parent.mkdir(parents=True, exist_ok=True)
            parsed_cache.write_text(
                _json.dumps(result.model_dump(), default=str),
                encoding="utf-8",
            )
        except Exception:
            pass  # non-critical — caching is best-effort

        return result

    def _try_llm_parsing(
        self, root: ET.Element, full_text: str, warnings: list[str]
    ) -> tuple[dict[str, Reference], list[Citation], list[str]] | None:
        """Try LLM-based reference parsing. Returns None if unavailable."""
        from src import config as app_config

        api_key = app_config.openai_api_key()
        if not api_key:
            return None

        # Extract raw ref text from GROBID XML
        raw_refs = self._extract_raw_ref_texts(root)
        if not raw_refs:
            return None

        # Extract citation markers from body
        markers = self._extract_citation_markers(root)

        # Call LLM using synchronous OpenAI client directly.
        # The GROBID parser's parse() is sync, so we avoid async entirely here.
        try:
            import httpx as _httpx
            from openai import OpenAI
            from src.parsers.llm_ref_parser import _SYSTEM_PROMPT, _build_prompt

            llm_config = app_config.llm()
            if not llm_config:
                return None

            api_key = app_config.openai_api_key()
            model = llm_config.get("model", "gpt-4o-mini")
            temperature = llm_config.get("temperature", 0.0)
            max_tokens = min(max(len(raw_refs) * 150 + 1000, 4096), 16384)

            timeout = _httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
            sync_client = OpenAI(api_key=api_key, timeout=timeout)

            import json as _json

            # Split into batches of ~30 refs to avoid API timeouts.
            # Each batch gets ALL markers (for matching), but only its refs.
            batch_size = 30
            llm_refs = []
            for i in range(0, len(raw_refs), batch_size):
                batch_refs = raw_refs[i:i + batch_size]
                # Renumber refs in batch starting from their original position
                numbered_refs = []
                for j, ref_text in enumerate(batch_refs):
                    numbered_refs.append((i + j + 1, ref_text))

                prompt_lines = [f"[{num}] {text}" for num, text in numbered_refs]
                batch_prompt = f"""## Raw References (from bibliography section)
{chr(10).join(prompt_lines)}

## Citation Markers (from body text)
{chr(10).join(markers)}

## Response Format
Respond in JSON:
{{"references": [
  {{"ref_num": 1, "title": "...", "authors": ["First Last"], "year": 2020, "venue": "...", "doi": null, "arxiv_id": null, "is_garbage": false, "matched_markers": ["(Smith et al., 2020)"]}}
]}}"""

                batch_max = min(max(len(batch_refs) * 150 + 1000, 4096), 16384)
                response = sync_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": batch_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=batch_max,
                    response_format={"type": "json_object"},
                )
                batch_data = _json.loads(response.choices[0].message.content or "{}")
                llm_refs.extend(batch_data.get("references", []))

            # Calculate cost
            cost = 0.0
            pricing = llm_config.get("pricing", {})
            if response.usage and pricing:
                cost = (
                    response.usage.prompt_tokens * pricing.get("input_cost_per_million", 0.15) / 1_000_000
                    + response.usage.completion_tokens * pricing.get("output_cost_per_million", 0.60) / 1_000_000
                )

            log.info(f"LLM parsed {len(llm_refs)} references, cost: ${cost:.4f}")
        except Exception as e:
            msg = str(e) or type(e).__name__
            warnings.append(f"LLM ref parsing failed: {msg}. Using GROBID fallback.")
            return None

        if not llm_refs:
            return None

        extra_warnings = [f"LLM ref-parsing cost: ${cost:.4f}"]

        # Build Reference objects from LLM output
        references: dict[str, Reference] = {}
        marker_to_ref: dict[str, str] = {}  # marker text → ref_id

        for llm_ref in llm_refs:
            ref_num = llm_ref.get("ref_num")
            if ref_num is None:
                continue

            ref_id = str(ref_num)
            title = (llm_ref.get("title") or "").strip()
            if title == "?":
                title = ""
            authors = llm_ref.get("authors") or []

            # Rule-based garbage check — don't trust LLM's is_garbage.
            # Keep anything with a title OR an author (blogs, tweets, etc.).
            # Only drop entries that are truly bare URLs or unintelligible.
            authors = [a.strip() for a in authors if a.strip()]
            has_title = bool(title) and len(title) > 3
            has_author = bool(authors) and any(len(a) > 1 for a in authors)
            if not has_title and not has_author:
                continue  # bare URL or unintelligible — genuinely garbage
            year = llm_ref.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            elif not isinstance(year, int):
                year = None

            venue = (llm_ref.get("venue") or "").strip() or None
            doi = (llm_ref.get("doi") or "").strip() or None
            arxiv_id = (llm_ref.get("arxiv_id") or "").strip() or None

            # Clean DOI: PDF line breaks create spaces in DOIs.
            # DOIs never contain spaces — always safe to remove them.
            if doi:
                doi = doi.replace(" ", "")

            # Extract URL from raw text.
            # PDF line breaks create two common artifacts:
            #   1. "https: //" (space after colon)
            #   2. "path- continuation" (hyphen + space at line break)
            # We fix BOTH before matching, so the URL is reconstructed properly.
            url = None
            if ref_num <= len(raw_refs):
                raw_for_url = raw_refs[ref_num - 1]
                # Fix "https: //" → "https://"
                raw_for_url = re.sub(r'(https?)\s*:\s*//', r'\1://', raw_for_url)
                # Fix "word- word" → "wordword" (rejoin hyphenated line breaks)
                raw_for_url = re.sub(r'-\s+', '', raw_for_url)
                url_match = re.search(r'https?://\S+', raw_for_url)
                if url_match:
                    url = url_match.group(0).rstrip('.,;)')

            raw_text = raw_refs[ref_num - 1] if ref_num <= len(raw_refs) else ""

            from src.citation.format_detector import detect_citation_format

            ref = Reference(
                ref_id=ref_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                arxiv_id=arxiv_id,
                url=url,
                raw_text=raw_text[:300],
                source_format="grobid+llm",
                citation_format=detect_citation_format(raw_text, "grobid+llm"),
            )
            references[ref_id] = ref

            # Build marker → ref_id mapping
            for marker in (llm_ref.get("matched_markers") or []):
                marker_to_ref[marker.strip()] = ref_id

        if not references:
            return None

        # Build Citation objects using GROBID's own target mapping (most reliable)
        citations, orphaned_bids = self._build_citations_from_grobid_targets_with_orphans(
            root, full_text, references, extra_warnings
        )

        # Recover orphaned refs: if body text cites a ref that was dropped,
        # the paper IS citing it — so it's not garbage. Recover it from
        # the GROBID XML raw text.
        if orphaned_bids and raw_refs:
            recovered = 0
            listbibl = root.find(".//tei:listBibl", TEI_NS)
            all_bibs = listbibl.findall(".//tei:biblStruct", TEI_NS) if listbibl is not None else []

            for bid in orphaned_bids:
                idx = next(
                    (i for i, b in enumerate(all_bibs)
                     if b.get("{http://www.w3.org/XML/1998/namespace}id") == bid),
                    None,
                )
                if idx is None or idx >= len(raw_refs):
                    continue

                ref_num = idx + 1
                ref_id = str(ref_num)
                if ref_id in references:
                    continue  # already exists

                raw = raw_refs[idx]

                # Only recover if raw text has substance (not a bare URL)
                is_bare_url = bool(re.match(r'^(URL\s+)?https?://\S+\.?$', raw.strip(), re.IGNORECASE))
                if is_bare_url or len(raw.strip()) < 15:
                    continue

                # Create a minimal Reference from raw text
                from src.citation.format_detector import detect_citation_format

                ref = Reference(
                    ref_id=ref_id,
                    title=None,  # let the agent figure it out from raw_text
                    authors=[],
                    year=None,
                    raw_text=raw[:300],
                    source_format="grobid+recovered",
                    citation_format=detect_citation_format(raw, "grobid+recovered"),
                )

                # Try to extract URL from raw text
                url_match = re.search(r'https?://\S+', raw)
                if url_match:
                    ref.url = url_match.group(0).rstrip('.,;)')

                references[ref_id] = ref
                recovered += 1

            if recovered:
                extra_warnings.append(f"Recovered {recovered} reference(s) from orphaned citations.")
                # Rebuild citations to include the recovered refs
                citations, _ = self._build_citations_from_grobid_targets_with_orphans(
                    root, full_text, references, extra_warnings
                )

        return references, citations, extra_warnings

    def _build_citations_from_grobid_targets_with_orphans(
        self,
        root: ET.Element,
        full_text: str,
        references: dict[str, Reference],
        warnings: list[str],
        bid_override: dict[str, str] | None = None,
    ) -> tuple[list[Citation], set[str]]:
        """Build citations and return orphaned biblio_ids.

        Same as _build_citations_from_grobid_targets but also returns the set
        of biblio_ids that are cited in the body but have no matching reference.
        Used for recovering wrongly-dropped refs.
        """
        citations = self._build_citations_from_grobid_targets(
            root, full_text, references, warnings, bid_override
        )

        # Find orphaned biblio_ids: cited in body but not in our references
        listbibl = root.find(".//tei:listBibl", TEI_NS)
        bid_to_refnum: dict[str, str] = {}
        if bid_override:
            bid_to_refnum = bid_override
        elif listbibl is not None:
            for i, bib in enumerate(listbibl.findall(".//tei:biblStruct", TEI_NS), 1):
                bid = bib.get("{http://www.w3.org/XML/1998/namespace}id")
                if bid:
                    bid_to_refnum[bid] = str(i)

        body = root.find(".//tei:body", TEI_NS)
        orphaned: set[str] = set()
        if body is not None:
            tei_uri = TEI_NS["tei"]
            for ref_elem in body.iter(f"{{{tei_uri}}}ref"):
                if ref_elem.get("type") == "bibr":
                    target = ref_elem.get("target", "")
                    for t in target.split():
                        bid = t.lstrip("#")
                        ref_num = bid_to_refnum.get(bid)
                        if ref_num and ref_num not in references:
                            orphaned.add(bid)

        return citations, orphaned

    @staticmethod
    def _extract_raw_ref_texts(root: ET.Element) -> list[str]:
        """Extract raw reference text from each biblStruct.

        Prefers note[@type="raw_reference"] — the actual text as it appears
        in the PDF (available when GROBID is called with includeRawCitations=1).
        Falls back to itertext() if raw_reference is not available.

        Using the raw PDF text is critical because itertext() is GROBID's
        already-parsed fields mashed together, which loses URLs, garbles DOIs,
        and mixes up author boundaries.
        """
        listbibl = root.find(".//tei:listBibl", TEI_NS)
        if listbibl is None:
            return []
        raw_refs = []
        for bib in listbibl.findall(".//tei:biblStruct", TEI_NS):
            # Prefer raw_reference (actual PDF text)
            note = bib.find('.//tei:note[@type="raw_reference"]', TEI_NS)
            if note is not None and note.text and len(note.text.strip()) > 10:
                text = note.text.strip()
            else:
                # Fallback to itertext (GROBID's parsed fields)
                text = " ".join(bib.itertext()).strip()
            text = re.sub(r"\s+", " ", text)
            raw_refs.append(text)  # no truncation — LLM needs the full text
        return raw_refs

    @staticmethod
    def _extract_citation_markers(root: ET.Element) -> list[str]:
        """Extract unique citation marker strings from body text."""
        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            return []
        markers: set[str] = set()
        tei_uri = TEI_NS["tei"]
        for ref in body.iter(f"{{{tei_uri}}}ref"):
            if ref.get("type") == "bibr":
                text = "".join(ref.itertext()).strip()
                if text and len(text) > 2:
                    markers.add(text)
        return sorted(markers)

    def _build_citations_from_grobid_targets(
        self,
        root: ET.Element,
        full_text: str,
        references: dict[str, Reference],
        warnings: list[str],
        bid_override: dict[str, str] | None = None,
    ) -> list[Citation]:
        """Build Citation objects using GROBID's own target mapping.

        GROBID's TEI XML has <ref target="#bN"> for each citation in the body,
        directly linking the marker to its bibliography entry. This is far more
        reliable than text-based marker matching (87% vs 54% coverage).

        Args:
            bid_override: Optional biblio_id → ref_id mapping. If None, uses
                sequential mapping (bN → "N+1") which works for LLM-parsed refs.
                If provided (from _extract_bibliography), uses the post-cleanup
                mapping where some ref_ids may be skipped.
        """
        if not full_text:
            return []

        # Build biblio_id → ref_id mapping
        if bid_override:
            bid_to_refnum = bid_override
        else:
            listbibl = root.find(".//tei:listBibl", TEI_NS)
            bid_to_refnum: dict[str, str] = {}
            if listbibl is not None:
                for i, bib in enumerate(listbibl.findall(".//tei:biblStruct", TEI_NS), 1):
                    bid = bib.get("{http://www.w3.org/XML/1998/namespace}id")
                    if bid:
                        bid_to_refnum[bid] = str(i)

        # Walk body to find citation markers with their text position
        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            return []

        citations: list[Citation] = []
        unresolved: set[str] = set()

        # Rebuild body text while tracking positions of <ref> elements
        # This mirrors _extract_text_and_citations but records citation info
        text_parts: list[str] = []
        citation_spots: list[dict] = []  # {ref_id, marker_text, char_position}
        tei_uri = TEI_NS["tei"]
        bibr_tag = f"{{{tei_uri}}}ref"
        structural_tags = frozenset((
            f"{{{tei_uri}}}div", f"{{{tei_uri}}}p", f"{{{tei_uri}}}head",
        ))

        stack: list[tuple[str, ET.Element]] = [("enter", body)]
        while stack:
            phase, elem = stack.pop()

            if phase == "tail":
                if elem.tail:
                    text_parts.append(elem.tail)
                continue

            if phase == "bibr":
                target = elem.get("target", "")
                marker_text = "".join(elem.itertext()).strip()
                char_pos = sum(len(p) for p in text_parts)

                if marker_text:
                    text_parts.append(marker_text)

                for t in target.split():
                    bid = t.lstrip("#")
                    ref_num = bid_to_refnum.get(bid)
                    if ref_num and ref_num in references:
                        citation_spots.append({
                            "ref_id": ref_num,
                            "marker": marker_text or f"[{ref_num}]",
                            "position": char_pos,
                        })
                    elif bid:
                        unresolved.add(bid)
                continue

            # "enter" phase
            if elem.tag in structural_tags:
                if text_parts and text_parts[-1] not in ("\n\n", "\n"):
                    text_parts.append("\n\n")

            if elem.text:
                text_parts.append(elem.text)

            for child in reversed(list(elem)):
                if child.tag == bibr_tag and child.get("type") == "bibr":
                    stack.append(("tail", child))
                    stack.append(("bibr", child))
                else:
                    stack.append(("tail", child))
                    stack.append(("enter", child))

        rebuilt_text = "".join(text_parts)
        rebuilt_text = re.sub(r" +", " ", rebuilt_text)
        rebuilt_text = re.sub(r"\n{3,}", "\n\n", rebuilt_text)

        # Build Citation objects from the recorded spots
        for spot in citation_spots:
            ctx = extract_context(rebuilt_text, spot["position"])
            citations.append(Citation(
                ref_id=spot["ref_id"],
                citing_sentence=ctx["citing_sentence"],
                context_before=ctx["context_before"],
                context_after=ctx["context_after"],
                marker=spot["marker"],
                position=spot["position"],
            ))

        if unresolved:
            warnings.append(
                f"{len(unresolved)} citation target(s) could not be matched "
                f"to references: {', '.join(sorted(unresolved)[:10])}"
            )

        return citations

    @staticmethod
    def _deduplicate_references(
        references: dict[str, Reference],
    ) -> tuple[dict[str, Reference], int]:
        """Remove references with duplicate titles, keeping the first occurrence."""
        seen_titles: set[str] = set()
        deduped: dict[str, Reference] = {}
        removed = 0
        for ref_id, ref in references.items():
            norm = _normalize_text(ref.title or "")
            if norm and norm in seen_titles:
                removed += 1
                continue
            if norm:
                seen_titles.add(norm)
            deduped[ref_id] = ref
        return deduped, removed

    # --- GROBID communication ---

    def _ensure_grobid(self) -> None:
        """Verify GROBID is reachable."""
        try:
            resp = self.session.get(
                f"{self.service_url}/api/isalive",
                timeout=self._health_check_timeout,
            )
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        raise GrobidError(
            f"GROBID not reachable at {self.service_url}. "
            f"Start it with: docker run --rm -p 8070:8070 {self._docker_image}"
        )

    def _process_pdf(self, pdf_path: str) -> str:
        """Send PDF to GROBID and return TEI XML.

        Retries on transient request errors with exponential backoff.
        Config is read at call time (not import time) for testability.
        """
        cfg = config.grobid()
        retryer = retry(
            stop=stop_after_attempt(cfg["retry_max_attempts"]),
            wait=wait_exponential(
                multiplier=cfg["retry_multiplier"],
                min=cfg["retry_min_wait"],
                max=cfg["retry_max_wait"],
            ),
            retry=retry_if_exception_type(requests.exceptions.RequestException),
        )

        @retryer
        def _do_request() -> str:
            url = f"{self.service_url}/api/processFulltextDocument"
            with open(pdf_path, "rb") as f:
                resp = self.session.post(
                    url,
                    files={"input": f},
                    data={
                        "consolidateCitations": "1",
                        "includeRawCitations": "1",
                    },
                    timeout=self.timeout,
                )
            if resp.status_code == 503:
                time.sleep(5)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.text

        return _do_request()

    # --- TEI XML extraction ---

    def _extract_paper_metadata(self, root: ET.Element) -> dict:
        """Extract paper-level title and authors from TEI header."""
        metadata: dict = {}
        title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        if title_elem is not None and title_elem.text:
            metadata["title"] = title_elem.text.strip()

        authors = []
        for author in root.findall(".//tei:fileDesc//tei:author", TEI_NS):
            persname = author.find(".//tei:persName", TEI_NS)
            if persname is not None:
                forename = persname.findtext("tei:forename", default="", namespaces=TEI_NS).strip()
                surname = persname.findtext("tei:surname", default="", namespaces=TEI_NS).strip()
                name = f"{forename} {surname}".strip()
                if name:
                    authors.append(name)
        if authors:
            metadata["authors"] = authors

        return metadata

    def _extract_bibliography(
        self, root: ET.Element, paper_title: str
    ) -> tuple[dict[str, Reference], dict[str, str]]:
        """Extract bibliography entries from TEI XML.

        Returns:
            references: dict mapping ref_id (str number) to Reference
            biblio_id_map: dict mapping GROBID biblio_id ('b0') to ref_id ('1')
        """
        references: dict[str, Reference] = {}
        biblio_id_map: dict[str, str] = {}
        ref_number = 1

        listbibl = root.find(".//tei:listBibl", TEI_NS)
        if listbibl is None:
            return references, biblio_id_map

        for biblstruct in listbibl.findall(".//tei:biblStruct", TEI_NS):
            biblio_id = biblstruct.get("{http://www.w3.org/XML/1998/namespace}id")
            if not biblio_id:
                continue

            ref_data = self._parse_biblstruct(biblstruct)
            if not _is_valid_reference(ref_data):
                continue
            ref_data = _clean_grobid_reference(ref_data)
            if ref_data is None:
                continue
            if paper_title and ref_data.get("title") and _is_self_reference(ref_data["title"], paper_title):
                continue

            ref_id = str(ref_number)
            from src.citation.format_detector import detect_citation_format

            raw_txt = ref_data.get("raw_text", "")
            ref = Reference(
                ref_id=ref_id,
                title=ref_data.get("title"),
                authors=ref_data.get("authors", []),
                year=ref_data.get("year"),
                venue=ref_data.get("venue"),
                doi=ref_data.get("doi"),
                arxiv_id=ref_data.get("arxiv_id"),
                url=ref_data.get("url"),
                raw_text=raw_txt,
                source_format="grobid",
                citation_format=detect_citation_format(raw_txt, "grobid"),
            )
            references[ref_id] = ref
            biblio_id_map[biblio_id] = ref_id
            ref_number += 1

        return references, biblio_id_map

    def _parse_biblstruct(self, elem: ET.Element) -> dict:
        """Parse a single <biblStruct> element into a dict."""
        # Title
        title = ""
        for path in [
            ".//tei:analytic/tei:title[@type='main']",
            ".//tei:monogr/tei:title[@type='main']",
            ".//tei:title",
        ]:
            te = elem.find(path, TEI_NS)
            if te is not None and te.text:
                title = te.text.strip()
                break
        # Clean garbage from title
        if title:
            title = re.sub(r'\s*arXiv preprint arXiv[:\s]*[\d.]+\s*$', '', title, flags=re.IGNORECASE)
            title = re.sub(r'//.*$', '', title).strip()

        # Year
        year: int | None = None
        for date_type in ["published", None]:
            xpath = f'.//tei:date[@type="{date_type}"]' if date_type else ".//tei:date"
            de = elem.find(xpath, TEI_NS)
            if de is not None:
                when = de.get("when", "")
                ym = re.search(r'^(\d{4})', when)
                if ym:
                    year = int(ym.group(1))
                    break
                if de.text:
                    ym = re.search(r'\b(\d{4})\b', de.text)
                    if ym:
                        year = int(ym.group(1))
                        break

        # Authors
        authors: list[str] = []
        for author_elem in elem.findall(".//tei:author", TEI_NS):
            pn = author_elem.find(".//tei:persName", TEI_NS)
            if pn is not None:
                forename = pn.findtext("tei:forename", default="", namespaces=TEI_NS).strip()
                surname = pn.findtext("tei:surname", default="", namespaces=TEI_NS).strip()
                name = f"{forename} {surname}".strip()
                if name:
                    authors.append(name)

        # Venue
        venue = ""
        for level in ["j", "m"]:
            ve = elem.find(f'.//tei:title[@level="{level}"]', TEI_NS)
            if ve is not None and ve.text:
                venue = ve.text.strip()
                break

        # Identifiers
        doi: str | None = None
        arxiv_id: str | None = None
        url: str | None = None
        for level_path in [".//tei:analytic", ".//tei:monogr"]:
            for idno in elem.findall(f"{level_path}/tei:idno", TEI_NS):
                id_type = (idno.get("type") or "").upper()
                id_val = (idno.text or "").strip()
                if id_type == "DOI" and id_val:
                    doi = id_val
                elif id_type == "ARXIV" and id_val:
                    arxiv_id = id_val
                elif id_type in ("URI", "URL") and id_val:
                    url = id_val
        # Also check for ptr elements with target URLs
        if not url:
            ptr = elem.find(".//tei:ptr", TEI_NS)
            if ptr is not None:
                target = ptr.get("target", "").strip()
                if target.startswith("http"):
                    url = target

        # Raw reference text
        raw_text = ""
        note = elem.find('.//tei:note[@type="raw_reference"]', TEI_NS)
        if note is not None and note.text:
            raw_text = note.text.strip()
        else:
            parts = []
            if authors:
                parts.append(", ".join(authors[:3]))
                if len(authors) > 3:
                    parts[-1] += " et al."
            if title:
                parts.append(title)
            if year:
                parts.append(f"({year})")
            raw_text = ". ".join(parts)

        # Extract URL from raw_text as fallback if not found in structured fields
        if not url and raw_text:
            url_match = re.search(r'https?://\S+', raw_text)
            if url_match:
                url = url_match.group(0).rstrip('.,;)')

        return {
            "title": title or None,
            "authors": authors,
            "year": year,
            "venue": venue or None,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": url,
            "raw_text": raw_text,
        }

    def _extract_text_and_citations(
        self, root: ET.Element, biblio_id_map: dict[str, str]
    ) -> tuple[str, set[str]]:
        """Extract body text and collect which references are cited inline.

        Uses an iterative tree walk with an explicit stack to avoid
        recursion-limit issues on deeply nested TEI XML.

        Returns (full_text, set of ref_ids cited in body).
        """
        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            return "", set()

        text_parts: list[str] = []
        cited_refs: set[str] = set()
        tei_uri = TEI_NS["tei"]

        structural_tags = frozenset((
            f"{{{tei_uri}}}div",
            f"{{{tei_uri}}}p",
            f"{{{tei_uri}}}head",
        ))
        bibr_tag = f"{{{tei_uri}}}ref"

        # Iterative tree walk using an explicit stack.
        # Each item is a tuple: (phase, element) where phase is one of:
        #   "enter" — process the element's own text and queue its children
        #   "tail"  — emit the element's tail text (text after closing tag)
        #   "bibr"  — handle an inline bibliography reference
        stack: list[tuple[str, ET.Element]] = [("enter", body)]

        while stack:
            phase, elem = stack.pop()

            if phase == "tail":
                if elem.tail:
                    text_parts.append(elem.tail)
                continue

            if phase == "bibr":
                target = elem.get("target", "")
                marker_text = "".join(elem.itertext()).strip()
                if marker_text:
                    text_parts.append(marker_text)
                for t in target.split():
                    bid = t.lstrip("#")
                    rid = biblio_id_map.get(bid)
                    if rid:
                        cited_refs.add(rid)
                continue

            # --- "enter" phase ---
            if elem.tag in structural_tags:
                if text_parts and text_parts[-1] not in ("\n\n", "\n"):
                    text_parts.append("\n\n")

            if elem.text:
                text_parts.append(elem.text)

            # Push children in reverse so the first child is processed first.
            for child in reversed(list(elem)):
                if child.tag == bibr_tag and child.get("type") == "bibr":
                    stack.append(("tail", child))
                    stack.append(("bibr", child))
                else:
                    stack.append(("tail", child))
                    stack.append(("enter", child))

        full_text = "".join(text_parts)
        full_text = re.sub(r" +", " ", full_text)
        full_text = re.sub(r"\n +", "\n", full_text)
        full_text = re.sub(r" +\n", "\n", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        return full_text.strip(), cited_refs

    @staticmethod
    def _resolve_key(key: str, references: dict[str, Reference]) -> Optional[str]:
        """Map a detected citation key to a reference ref_id.

        Numbered keys ('5') map directly. Author-year keys need matching.
        """
        # Direct numbered match
        if key in references:
            return key

        # Author-year: match against reference authors + year
        for ref_id, ref in references.items():
            if ref.year and str(ref.year) in key:
                for author in ref.authors:
                    surname = author.split()[-1] if author else ""
                    if normalize_author_name(surname) in key:
                        return ref_id

        return None
