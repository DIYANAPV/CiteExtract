"""Citation recovery layer for GROBID-parsed PDFs.

Why this exists
---------------
GROBID's CRF body NER has known recall gaps that the user observes as
"missing comment-boxes" on the annotated PDF:

1. **Truly orphan bibrs** — ``<ref type="bibr">`` with no ``target``
   attribute at all. Currently dropped silently by the body walk.
2. **Truncated multi-cites** — GROBID emits ``(Liang et al., 2024;`` as
   one bibr but never emits a second bibr for the trailing
   ``Zhuang et al., 2025)`` half.
3. **Citations to dropped refs** — body bibrs whose ``target`` points to
   a biblStruct GROBID excluded from ``listBibl``.

This module sits between the body XML walk and the linker. It takes the
bibrs GROBID emitted (resolved + orphan), runs the existing regex
``CitationDetector`` over the body text to catch what GROBID missed, and
merges them by character-span overlap. Candidates that don't link to any
reference are dropped — that's how venue-only mentions like
``(NeurIPS 2022)`` are filtered out.

Annotator and downstream verification are unchanged: this layer just
puts more (correctly-linked) ``Citation`` objects into ``ParsedPaper``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.citation.context_extractor import extract_context
from src.citation.detector import CitationDetector
from src.models.citation import Citation
from src.models.reference import Reference
from src.parsers.marker_rule_check import compute_rule_ref_id

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BibrSpan:
    """A single ``<ref type="bibr">`` span lifted from GROBID's body XML.

    ``target_ref_id`` is the resolved ``Reference.ref_id`` when GROBID's
    ``target`` attribute pointed to a known biblStruct AND that biblStruct
    survived ref deduplication. ``None`` for orphans — i.e. truncated
    multi-cites, bibrs with no ``target``, or bibrs whose ``target`` points
    to a dropped biblStruct.
    """

    position: int          # char offset in body_text
    marker: str            # text content of the bibr element
    target_ref_id: Optional[str]

    @property
    def end(self) -> int:
        return self.position + len(self.marker)


# ---------------------------------------------------------------------------
# Internal candidate model + dedup priority
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    """A pending Citation before linking + dedup.

    ``source`` records which extraction stage produced this candidate so
    we can deterministically pick a winner on dedup tiebreaks.
    """

    position: int
    marker: str
    end: int
    ref_id: Optional[str]
    source: str  # "grobid_resolved" | "grobid_orphan" | "regex"


# Higher index = more authoritative. Resolved GROBID wins because the
# ``target`` XML attribute is GROBID's own structured ground truth; orphan
# linking and regex linking both rely on author+year heuristics.
_SOURCE_PRIORITY = {
    "regex": 0,
    "grobid_orphan": 1,
    "grobid_resolved": 2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spans_overlap(a_pos: int, a_end: int, b_pos: int, b_end: int) -> bool:
    """True iff char ranges ``[a_pos, a_end)`` and ``[b_pos, b_end)`` intersect."""
    return a_pos < b_end and b_pos < a_end


# Characters allowed in the gap between two candidates that we still want
# to dedup as a single on-page appearance: the punctuation that separates
# items inside one citation parenthetical or comma-list. Notably *excludes*
# the period — a period implies a sentence boundary, which makes two
# same-ref citations distinct prose mentions, not one repeated marker.
_INTRA_PAREN_GAP_CHARS = frozenset(" \t\n,;")

# Max characters of gap between two same-ref candidates to still treat as
# one on-page appearance. Tuned for the longest realistic separator chain
# inside a multi-cite parenthetical: ``"; "`` after a closing-paren-less
# marker, with whitespace tolerance.
_MAX_INTRA_PAREN_GAP = 8


def _same_appearance(
    a_pos: int, a_end: int, b_pos: int, b_end: int, body_text: str,
) -> bool:
    """True iff two char spans plausibly cover the same on-page citation.

    Strictly stronger than ``_spans_overlap``: also returns True when the
    spans don't touch but the text between them is intra-parenthetical
    punctuation only. Catches GROBID-truncated multi-cite cases like
    ``(Foo, 2024, Yamada et al., 2025)`` where GROBID emits one bibr that
    ends mid-parenthetical and a second bibr that starts mid-parenthetical
    — distinct char spans, but the same physical comment-box anchor on
    the rendered PDF.

    Sentence boundaries (a period in the gap) keep the spans distinct;
    that's the discriminator between "one parenthetical with two refs"
    (collapse) and "two consecutive sentences citing the same ref"
    (keep both).
    """
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
    """Reconstruct a parseable marker string from a normalized regex key.

    ``CitationDetector`` emits keys like ``smithetal2020`` or ``smith2020a``
    by collapsing surname + ``etal`` + year. The linker
    (``compute_rule_ref_id``) wants a human-readable marker; this puts the
    pieces back so ``parse_marker_surface`` can pull surname + year out.

    Falls back to the raw key when the format is unexpected; the linker
    will then return ``None`` and the candidate gets dropped, which is the
    correct behavior for an unparseable key.
    """
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
    """Heuristic link of an orphan bibr's marker text to a Reference.

    Wraps ``compute_rule_ref_id`` so the recovery layer has a single name
    to call regardless of which path produced the marker.
    """
    return compute_rule_ref_id(marker, references, refs_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryStats:
    """Per-call summary of what each recovery path contributed.

    Surfaced in ``ParsedPaper.warnings`` so users can see which engine
    produced each citation when investigating a misplaced annotation.
    """

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
    """Merge GROBID bibrs with a regex sweep into a deduped Citation list.

    Args:
        body_text: GROBID's reconstructed body text. Must match the
            ``position`` field of each ``BibrSpan`` (i.e. positions are
            character offsets into this exact string).
        bibrs: All bibrs from the body XML walk, in document order.
            Resolved bibrs carry a ``target_ref_id``; orphans pass ``None``.
        references: ``{ref_id: Reference}``, the deduped reference list.

    Returns:
        ``(citations, stats)`` — ``citations`` sorted by position;
        ``stats`` for logging / surfacing in warnings.
    """
    refs_list = list(references.values())
    candidates: list[_Candidate] = []
    stats = {
        "grobid_resolved": 0,
        "grobid_orphan_linked": 0,
        "grobid_orphan_dropped": 0,
        "regex_added": 0,
        "regex_dropped": 0,
    }

    # --- Step 1: GROBID resolved bibrs (already linked via target attr) ---
    for b in bibrs:
        if b.target_ref_id and b.target_ref_id in references:
            candidates.append(_Candidate(
                position=b.position, marker=b.marker, end=b.end,
                ref_id=b.target_ref_id, source="grobid_resolved",
            ))
            stats["grobid_resolved"] += 1

    # --- Step 2: GROBID orphan bibrs — heuristic link by author+year ---
    # An orphan is a bibr whose ``target_ref_id`` is None *or* whose target
    # points to a ref that didn't survive dedup. Both look the same here.
    for b in bibrs:
        if b.target_ref_id and b.target_ref_id in references:
            continue  # handled by step 1
        ref_id = _link_orphan(b.marker, references, refs_list)
        if ref_id is not None:
            candidates.append(_Candidate(
                position=b.position, marker=b.marker, end=b.end,
                ref_id=ref_id, source="grobid_orphan",
            ))
            stats["grobid_orphan_linked"] += 1
        else:
            stats["grobid_orphan_dropped"] += 1

    # --- Step 3: Regex sweep over body text ---
    # Each DetectedCitation may carry multiple keys (multi-cite). Link each
    # key independently so ``(Liang, 2024; Zhuang, 2025)`` becomes two
    # candidates that can dedup separately against any GROBID bibrs.
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

    # --- Step 4: Dedup candidates that point to the same on-page citation ---
    # Sort by position so we emit citations in reading order, then collapse
    # same-ref candidates whose spans either overlap or are separated only
    # by intra-parenthetical punctuation (one comment-box anchor on the PDF).
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
            # Promote: replace the kept candidate with this more-trusted one.
            deduped[winner_idx] = cand

    # --- Step 5: Build Citation objects with citing-sentence context ---
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
