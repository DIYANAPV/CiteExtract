"""System prompt, verdict JSON schema, and user message builder for agentic verification.

The system prompt encodes domain expertise as guidance (not rigid rules),
allowing the LLM agent to reason about edge cases better than thresholds.
"""

from typing import Any

from src.models.citation import Citation
from src.models.reference import Reference
from src.verification.filters import is_substantive_citation


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Verify whether a bibliographic reference points to a real paper with \
correct metadata. When citing contexts are provided, also check whether \
the citing sentences accurately represent the cited work.

## How to search

DOI present → search_crossref_by_doi first (authoritative).
arXiv ID present → search_semantic_scholar with search_type="id", \
query="ARXIV:<id>".
Title only → try search_semantic_scholar, then search_openalex (broadest, \
250M+ works), then search_pubmed (biomedical). Stop once you have a \
confident match.

For web sources (blogs, reports) not found in scholarly databases → \
verify_web_source to check the URL.

## How to compare metadata

Once you find a match, run compare_titles and compare_authors. Read the \
actual differences — do not just look at the similarity score.

Normal differences (not fabrication):
- Subtitle truncated, preprint vs published venue, year ±1
- Author name format variants, "et al." truncation
- Venue abbreviations ("NeurIPS" = "NIPS")

Suspicious differences:
- Title matches but completely different authors
- Year off by 2+ outside a preprint gap
- DOI resolves to a different paper
- Fields from different papers mixed together (blended reference)

When venue or year differs, search another database for the version the \
reference claims before concluding mismatch. When in doubt, search.

## How to check citing claims

If citing contexts are provided, use retrieve_cited_paper_passages to \
fetch passages from the cited paper. Compare the citing sentence against \
retrieved passages. Focus on what THIS specific citation claims — in a \
multi-citation sentence, each marker covers only its adjacent claim.

## Verdicts

**FABRICATED** — not found anywhere, retracted, or metadata from different \
papers blended together. Author truncation and venue abbreviations are \
not fabrication.

**MISREPRESENTED** — paper exists and metadata is acceptable, but the \
citing sentence contradicts what the paper actually says. Only assign \
after retrieving and checking passages.

**VALID** — exists and verifiable. Metadata correct or cosmetically different.

**UNVERIFIABLE** — insufficient data to determine (no title, all APIs \
failed).

Actions: FABRICATED → remove_citation, MISREPRESENTED → verify_claim, \
VALID/UNVERIFIABLE → no_action.

## Output rules

- Explain reasoning in the explanation field.
- List databases you searched in databases_checked.
- Report ALL discrepancies in field_discrepancies, even when verdict is \
  VALID — what the reference says, what the database says, whether you \
  investigated, your conclusion.
- Stop calling tools once you have enough evidence."""


# ---------------------------------------------------------------------------
# Verdict JSON schema for structured output
# ---------------------------------------------------------------------------

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "citation_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": [
                        "FABRICATED", "MISREPRESENTED", "VALID", "UNVERIFIABLE",
                    ],
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "no_action", "verify_claim", "remove_citation",
                    ],
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed reasoning for the verdict.",
                },
                "flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any concerns, anomalies, or notes for human review.",
                },
                "databases_checked": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of databases actually queried.",
                },
                "citation_format": {
                    "type": ["string", "null"],
                    "description": "Detected bibliography format (apa, vancouver, ieee, etc.).",
                },
                "matched_title": {
                    "type": ["string", "null"],
                    "description": "Title of the best matching paper found, or null.",
                },
                "matched_authors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Authors of the best matching paper found.",
                },
                "matched_year": {
                    "type": ["integer", "null"],
                    "description": "Year of the best matching paper found, or null.",
                },
                "matched_venue": {
                    "type": ["string", "null"],
                    "description": "Venue of the best matching paper found, or null.",
                },
                "matched_doi": {
                    "type": ["string", "null"],
                    "description": "DOI of the best matching paper found, or null.",
                },
                "abstract": {
                    "type": ["string", "null"],
                    "description": "Abstract of the matched paper (first 1500 chars), or null.",
                },
                "title_comparison": {
                    "type": ["object", "null"],
                    "description": "Structured title comparison result.",
                    "properties": {
                        "exact_match": {"type": "boolean"},
                        "differences": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "similarity": {"type": ["number", "null"]},
                    },
                    "required": ["exact_match", "differences", "similarity"],
                    "additionalProperties": False,
                },
                "author_comparison": {
                    "type": ["object", "null"],
                    "description": "Structured author comparison result.",
                    "properties": {
                        "matched_count": {"type": "integer"},
                        "unmatched_count": {"type": "integer"},
                        "unmatched_authors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "truncation_expected": {"type": "boolean"},
                        "explanation": {"type": ["string", "null"]},
                    },
                    "required": [
                        "matched_count", "unmatched_count", "unmatched_authors",
                        "truncation_expected", "explanation",
                    ],
                    "additionalProperties": False,
                },
                "is_retracted": {
                    "type": "boolean",
                    "description": "Whether the paper is retracted (from CrossRef).",
                },
                "field_discrepancies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "description": "Field name: title, authors, year, venue, doi.",
                            },
                            "ref_value": {
                                "type": ["string", "null"],
                                "description": "Value from the reference.",
                            },
                            "db_value": {
                                "type": ["string", "null"],
                                "description": "Value from the database.",
                            },
                            "status": {
                                "type": "string",
                                "description": "match, close_match, mismatch, or not_checked.",
                            },
                            "investigation_notes": {
                                "type": ["string", "null"],
                                "description": "What you investigated and concluded.",
                            },
                        },
                        "required": [
                            "field", "ref_value", "db_value", "status",
                            "investigation_notes",
                        ],
                        "additionalProperties": False,
                    },
                    "description": "Per-field comparison with investigation notes.",
                },
            },
            "required": [
                "verdict", "action", "explanation", "flags",
                "databases_checked", "citation_format",
                "matched_title", "matched_authors",
                "matched_year", "matched_venue", "matched_doi", "abstract",
                "title_comparison", "author_comparison",
                "is_retracted", "field_discrepancies",
            ],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

MAX_CITING_CONTEXTS = 3


def build_user_message(
    reference: Reference,
    citations: list[Citation],
    has_body_text: bool,
) -> str:
    """Build the per-reference user message for the agent.

    Includes reference metadata and (if available) citing contexts.
    """
    parts: list[str] = []

    parts.append("## Reference to Verify")
    parts.append(f"- **Title**: {reference.title or '(not available)'}")
    if reference.authors:
        parts.append(f"- **Authors**: {', '.join(reference.authors)}")
    if reference.year:
        parts.append(f"- **Year**: {reference.year}")
    if reference.venue:
        parts.append(f"- **Venue**: {reference.venue}")
    if reference.doi:
        parts.append(f"- **DOI**: {reference.doi}")
        parts.append(
            f"  → **This reference has a DOI. Start by looking it up with "
            f"search_crossref_by_doi(\"{reference.doi}\").**"
        )
    if reference.arxiv_id:
        parts.append(f"- **arXiv ID**: {reference.arxiv_id}")
    if getattr(reference, 'url', None):
        parts.append(f"- **URL**: {reference.url}")
        parts.append(
            f"  → **If not found in scholarly databases, use "
            f"verify_web_source(\"{reference.url}\") to check if this "
            f"web source exists.**"
        )

    # Citation format info
    fmt = getattr(reference, 'citation_format', None)
    if fmt:
        parts.append(f"- **Citation Format**: {fmt}")
        parts.append(
            f"  → Pass citation_format=\"{fmt}\" to compare_authors "
            f"for format-aware truncation analysis."
        )
    else:
        parts.append("- **Citation Format**: (not detected)")

    parts.append(f"- **Raw text**: {reference.raw_text[:300]}")

    # Add citing contexts if available
    substantive = [c for c in citations if is_substantive_citation(c)]
    if has_body_text and substantive:
        parts.append("")
        parts.append("## Citing Contexts (how this reference is used in the paper)")
        for i, cit in enumerate(substantive[:MAX_CITING_CONTEXTS], 1):
            context = _format_context(cit)
            parts.append(f"\n### Context {i}")
            parts.append(context)
    elif has_body_text:
        parts.append("")
        parts.append(
            "## Citing Contexts\n"
            "No substantive citing contexts found for this reference "
            "(only trivial mentions like 'see [X]'). Skip semantic verification."
        )
    else:
        parts.append("")
        parts.append(
            "## Citing Contexts\n"
            "No body text available (input was a bibliography file). "
            "Skip semantic verification — only check existence and metadata."
        )

    parts.append("")
    parts.append("## Your Task")
    parts.append(
        "Search scholarly databases to verify this reference. Check existence, "
        "metadata accuracy, and (if citing contexts and abstract are available) "
        "semantic accuracy. Then provide your verdict."
    )

    return "\n".join(parts)


def _format_context(citation: Citation) -> str:
    """Format a citation's context window (paragraph-bounded, configurable)."""
    sentences = []
    if citation.context_before:
        sentences.append(citation.context_before.strip())
    sentences.append(f"**{citation.citing_sentence.strip()}**")
    if citation.context_after:
        sentences.append(citation.context_after.strip())
    return " ".join(sentences)
