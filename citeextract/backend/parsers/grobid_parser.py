
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from defusedxml.ElementTree import fromstring as _safe_fromstring

log = logging.getLogger(__name__)

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from citeextract import config, paths
from citeextract.citation.detector import normalize_author_name
from citeextract.models.citation import Citation
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.reference import Reference
from citeextract.parsers.base import BaseParser
from citeextract.parsers.marker_rule_check import (
    annotate_citation_confidence,
    normalize_marker_key,
)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidError(Exception):
    pass


def _normalize_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def _is_self_reference(ref_title: str, paper_title: str) -> bool:
    from difflib import SequenceMatcher
    ref_norm = _normalize_text(ref_title)
    paper_norm = _normalize_text(paper_title)
    if not ref_norm or not paper_norm:
        return False
    if paper_norm in ref_norm:
        return True
    return SequenceMatcher(None, ref_norm, paper_norm).ratio() > config.thresholds()["title_match"]


def _is_valid_reference(ref_data: dict) -> bool:
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
    title = ref_data.get("title") or ""


    if re.match(r'^(Fig\.|Figure|Table|Plate)\s', title, re.IGNORECASE):
        return None

    if len(title) > 150:
        casual = re.search(
            r'\b(scary|amazing|cool|weird|funny|basically|stuff|'
            r'things like|gonna|wanna|crazy|lol|btw)\b',
            title, re.IGNORECASE,
        )
        if casual:
            return None


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
        import hashlib
        import json as _json
        pdf_bytes = Path(file_path).read_bytes()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

        parsed_cache = paths.data_dir() / "cache" / f"parsed_{pdf_hash}.json"
        if parsed_cache.exists():
            try:
                cached_data = _json.loads(parsed_cache.read_text(encoding="utf-8"))
                return ParsedPaper(**cached_data)
            except Exception:
                pass

        cache_path = paths.data_dir() / "cache" / f"grobid_{pdf_hash}.xml"
        if cache_path.exists():
            xml_content = cache_path.read_text(encoding="utf-8")
        else:
            self._ensure_grobid()
            xml_content = self._process_pdf(file_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(xml_content, encoding="utf-8")

        root = _safe_fromstring(xml_content)

        warnings: list[str] = []

        metadata = self._extract_paper_metadata(root)
        paper_title = metadata.get("title", "")

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

        references_dict, biblio_id_map_for_rebuild = self._extract_bibliography(
            root, paper_title,
        )

        citations = self._build_citations_from_grobid_targets(
            root, full_text, references_dict, warnings,
            bid_override=biblio_id_map_for_rebuild,
        )
        citations = annotate_citation_confidence(citations, references_dict)

        citations = self._maybe_complete_bibliography(
            file_path, root, full_text, references_dict, citations,
            biblio_id_map_for_rebuild, warnings,
        )

        references_dict, dedup_remap = self._deduplicate_references(references_dict)
        if dedup_remap:
            for cit in citations:
                if cit.ref_id in dedup_remap:
                    cit.ref_id = dedup_remap[cit.ref_id]
            warnings.append(
                f"Deduplicated {len(dedup_remap)} reference(s) with identical "
                f"titles; remapped citations to kept refs."
            )

        result = ParsedPaper(
            references=list(references_dict.values()),
            citations=citations,
            has_body_text=bool(full_text),
            body_text=full_text or "",
            input_format="pdf",
            metadata=metadata,
            warnings=warnings,
        )

        try:
            parsed_cache.parent.mkdir(parents=True, exist_ok=True)
            parsed_cache.write_text(
                _json.dumps(result.model_dump(), default=str),
                encoding="utf-8",
            )
        except (OSError, TypeError) as e:
            log.debug(f"ParsedPaper cache write failed: {type(e).__name__}: {e}")

        return result

    def _maybe_complete_bibliography(
        self,
        file_path: str,
        root: ET.Element,
        full_text: str,
        references_dict: dict[str, Reference],
        citations: list[Citation],
        biblio_id_map: dict[str, str] | None,
        warnings: list[str],
    ) -> list[Citation]:
        from citeextract import config as app_config
        from citeextract.parsers.bibliography_completion import (
            assign_ref_ids,
            complete_bibliography_sync,
            should_complete,
        )

        recovery_stats = getattr(self, "_last_recovery_stats", None)
        if recovery_stats is None:
            return citations

        if not should_complete(
            recovery_stats.grobid_orphan_dropped,
            recovery_stats.regex_dropped,
            len(references_dict),
        ):
            return citations

        api_key = app_config.openai_api_key()
        if not api_key:
            warnings.append(
                "Bibliography completion skipped — set OPENAI_API_KEY to "
                f"recover the {recovery_stats.grobid_orphan_dropped + recovery_stats.regex_dropped} "
                "candidate(s) that didn't link to any reference."
            )
            return citations

        llm_config = app_config.llm() or {}
        log.info(
            "phase3: triggering bibliography completion "
            "(orphan_dropped=%d, regex_dropped=%d, refs=%d)",
            recovery_stats.grobid_orphan_dropped,
            recovery_stats.regex_dropped,
            len(references_dict),
        )

        extra_refs, cost = complete_bibliography_sync(
            file_path, references_dict, llm_config, api_key,
        )
        if not extra_refs:
            return citations

        assign_ref_ids(extra_refs, references_dict)
        for ref in extra_refs:
            references_dict[ref.ref_id] = ref

        citations = self._build_citations_from_grobid_targets(
            root, full_text, references_dict, warnings,
            bid_override=biblio_id_map,
        )
        citations = annotate_citation_confidence(citations, references_dict)

        warnings.append(
            f"Bibliography completion: recovered {len(extra_refs)} reference(s) "
            f"GROBID dropped (cost ${cost:.4f}); citations re-linked."
        )
        return citations

    def _build_citations_from_grobid_targets(
        self,
        root: ET.Element,
        full_text: str,
        references: dict[str, Reference],
        warnings: list[str],
        bid_override: dict[str, str] | None = None,
    ) -> list[Citation]:
        if not full_text:
            return []

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

        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            return []

        unresolved: set[str] = set()

        from citeextract.parsers.citation_recovery import BibrSpan, recover_citations

        text_parts: list[str] = []
        bibr_spans: list[BibrSpan] = []
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

                emitted_any = False
                for t in target.split():
                    bid = t.lstrip("#")
                    ref_num = bid_to_refnum.get(bid)
                    if ref_num and ref_num in references:
                        bibr_spans.append(BibrSpan(
                            position=char_pos,
                            marker=marker_text or f"[{ref_num}]",
                            target_ref_id=ref_num,
                        ))
                        emitted_any = True
                    elif bid:
                        unresolved.add(bid)

                if not emitted_any and marker_text:
                    bibr_spans.append(BibrSpan(
                        position=char_pos,
                        marker=marker_text,
                        target_ref_id=None,
                    ))
                continue

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

        citations, recovery_stats = recover_citations(
            rebuilt_text, bibr_spans, references,
        )
        self._last_recovery_stats = recovery_stats

        recovered = (
            recovery_stats.grobid_orphan_linked + recovery_stats.regex_added
        )
        if recovered > 0:
            warnings.append(
                f"Citation recovery: GROBID resolved "
                f"{recovery_stats.grobid_resolved}; rescued "
                f"{recovery_stats.grobid_orphan_linked} orphan bibr(s); "
                f"regex sweep added {recovery_stats.regex_added} candidate(s); "
                f"{recovery_stats.final} total after dedup."
            )

        if unresolved:
            warnings.append(
                f"{len(unresolved)} citation target(s) could not be matched "
                f"to references: {', '.join(sorted(unresolved)[:10])}"
            )

        return citations

    @staticmethod
    def _deduplicate_references(
        references: dict[str, Reference],
    ) -> tuple[dict[str, Reference], dict[str, str]]:
        seen_titles: dict[str, str] = {}
        deduped: dict[str, Reference] = {}
        remap: dict[str, str] = {}
        for ref_id, ref in references.items():
            norm = _normalize_text(ref.title or "")
            if norm and norm in seen_titles:
                remap[ref_id] = seen_titles[norm]
                continue
            if norm:
                seen_titles[norm] = ref_id
            deduped[ref_id] = ref
        return deduped, remap


    def _ensure_grobid(self) -> None:
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


    def _extract_paper_metadata(self, root: ET.Element) -> dict:
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
            from citeextract.citation.format_detector import detect_citation_format

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
        if title:
            title = re.sub(r'\s*arXiv preprint arXiv[:\s]*[\d.]+\s*$', '', title, flags=re.IGNORECASE)
            title = re.sub(r'//.*$', '', title).strip()

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

        authors: list[str] = []
        for author_elem in elem.findall(".//tei:author", TEI_NS):
            pn = author_elem.find(".//tei:persName", TEI_NS)
            if pn is not None:
                forename = pn.findtext("tei:forename", default="", namespaces=TEI_NS).strip()
                surname = pn.findtext("tei:surname", default="", namespaces=TEI_NS).strip()
                name = f"{forename} {surname}".strip()
                if name:
                    authors.append(name)

        venue = ""
        for level in ["j", "m"]:
            ve = elem.find(f'.//tei:title[@level="{level}"]', TEI_NS)
            if ve is not None and ve.text:
                venue = ve.text.strip()
                break

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
        if not url:
            ptr = elem.find(".//tei:ptr", TEI_NS)
            if ptr is not None:
                target = ptr.get("target", "").strip()
                if target.startswith("http"):
                    url = target

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

        full_text = "".join(text_parts)
        full_text = re.sub(r" +", " ", full_text)
        full_text = re.sub(r"\n +", "\n", full_text)
        full_text = re.sub(r" +\n", "\n", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        return full_text.strip(), cited_refs

    @staticmethod
    def _resolve_key(key: str, references: dict[str, Reference]) -> Optional[str]:
        if key in references:
            return key

        for ref_id, ref in references.items():
            if ref.year and str(ref.year) in key:
                for author in ref.authors:
                    surname = author.split()[-1] if author else ""
                    if normalize_author_name(surname) in key:
                        return ref_id

        return None
