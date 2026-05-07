
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from citeextract.citation.context_extractor import extract_context
from citeextract.citation.detector import CitationDetector
from citeextract.models.citation import Citation
from citeextract.models.reference import Reference
from citeextract.parsers.marker_rule_check import compute_rule_ref_id

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BibrSpan:

    position: int
    marker: str
    target_ref_id: Optional[str]

    @property
    def end(self) -> int:
        return self.position + len(self.marker)


@dataclass
class _Candidate:

    position: int
    marker: str
    end: int
    ref_id: Optional[str]
    source: str


_SOURCE_PRIORITY = {
    "regex": 0,
    "grobid_orphan": 1,
    "grobid_resolved": 2,
}


def _spans_overlap(a_pos: int, a_end: int, b_pos: int, b_end: int) -> bool:
    return a_pos < b_end and b_pos < a_end


_INTRA_PAREN_GAP_CHARS = frozenset(" \t\n,;")

_MAX_INTRA_PAREN_GAP = 8


def _same_appearance(
    a_pos: int, a_end: int, b_pos: int, b_end: int, body_text: str,
) -> bool:
    if _spans_overlap(a_pos, a_end, b_pos, b_end):
        return True
    lo_end = min(a_end, b_end)
    hi_start = max(a_pos, b_pos)
    gap_size = hi_start - lo_end
    if gap_size <= 0 or gap_size > _MAX_INTRA_PAREN_GAP:
        return False
    between = body_text[lo_end:hi_start]
    return all(c in _INTRA_PAREN_GAP_CHARS for c in between)


_KEY_RE = re.compile(r"^(.+?)(etal)?(\d{4}[a-z]?)$")


def _key_to_synthetic_marker(key: str) -> str:
    m = _KEY_RE.match(key)
    if not m:
        return key
    surname, etal, year = m.group(1), m.group(2), m.group(3)
    parts = [surname.capitalize()]
    if etal:
        parts.append("et al.")
    parts.append(year)
    return " ".join(parts)


def _link_orphan(
    marker: str,
    references: dict[str, Reference],
    refs_list: list[Reference],
) -> Optional[str]:
    return compute_rule_ref_id(marker, references, refs_list)


@dataclass(frozen=True)
class RecoveryStats:

    grobid_resolved: int = 0
    grobid_orphan_linked: int = 0
    grobid_orphan_dropped: int = 0
    regex_added: int = 0
    regex_dropped: int = 0
    final: int = 0


def recover_citations(
    body_text: str,
    bibrs: list[BibrSpan],
    references: dict[str, Reference],
) -> tuple[list[Citation], RecoveryStats]:
    refs_list = list(references.values())
    candidates: list[_Candidate] = []
    stats = {
        "grobid_resolved": 0,
        "grobid_orphan_linked": 0,
        "grobid_orphan_dropped": 0,
        "regex_added": 0,
        "regex_dropped": 0,
    }

    for b in bibrs:
        if b.target_ref_id and b.target_ref_id in references:
            candidates.append(_Candidate(
                position=b.position, marker=b.marker, end=b.end,
                ref_id=b.target_ref_id, source="grobid_resolved",
            ))
            stats["grobid_resolved"] += 1

    for b in bibrs:
        if b.target_ref_id and b.target_ref_id in references:
            continue
        ref_id = _link_orphan(b.marker, references, refs_list)
        if ref_id is not None:
            candidates.append(_Candidate(
                position=b.position, marker=b.marker, end=b.end,
                ref_id=ref_id, source="grobid_orphan",
            ))
            stats["grobid_orphan_linked"] += 1
        else:
            stats["grobid_orphan_dropped"] += 1

    for hit in CitationDetector().detect_all(body_text):
        for key in hit.keys:
            synthetic = _key_to_synthetic_marker(key)
            ref_id = compute_rule_ref_id(synthetic, references, refs_list)
            if ref_id is None:
                stats["regex_dropped"] += 1
                continue
            candidates.append(_Candidate(
                position=hit.position, marker=hit.marker,
                end=hit.position + len(hit.marker),
                ref_id=ref_id, source="regex",
            ))
            stats["regex_added"] += 1

    candidates.sort(key=lambda c: (c.position, c.ref_id or "",
                                   -_SOURCE_PRIORITY[c.source]))
    deduped: list[_Candidate] = []
    for cand in candidates:
        winner_idx: Optional[int] = None
        for i, kept in enumerate(deduped):
            if (kept.ref_id == cand.ref_id
                    and _same_appearance(kept.position, kept.end,
                                         cand.position, cand.end,
                                         body_text)):
                winner_idx = i
                break
        if winner_idx is None:
            deduped.append(cand)
            continue
        kept = deduped[winner_idx]
        if _SOURCE_PRIORITY[cand.source] > _SOURCE_PRIORITY[kept.source]:
            deduped[winner_idx] = cand

    out: list[Citation] = []
    for c in deduped:
        ctx = extract_context(body_text, c.position)
        out.append(Citation(
            ref_id=c.ref_id,  # type: ignore[arg-type]  # None filtered above
            citing_sentence=ctx["citing_sentence"],
            context_before=ctx["context_before"],
            context_after=ctx["context_after"],
            marker=c.marker,
            position=c.position,
        ))

    out.sort(key=lambda c: c.position)
    rstats = RecoveryStats(final=len(out), **stats)
    log.info(
        "citation_recovery: resolved=%d orphan_linked=%d orphan_dropped=%d "
        "regex_added=%d regex_dropped=%d final=%d",
        rstats.grobid_resolved, rstats.grobid_orphan_linked,
        rstats.grobid_orphan_dropped, rstats.regex_added,
        rstats.regex_dropped, rstats.final,
    )
    return out, rstats
