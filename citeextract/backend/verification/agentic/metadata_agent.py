
import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from citeextract.verification.api_clients.llm_client import CostTracker
from citeextract.verification.agentic.tools import TOOL_DEFINITIONS, ToolExecutor

log = logging.getLogger(__name__)


METADATA_SYSTEM_PROMPT = """\
Investigate whether metadata mismatches between a reference and a \
database record are cosmetic or real.

You receive pre-computed comparison results (title diff, author match \
status). Read the actual differences before deciding whether to search.

## Cosmetic vs real

These are normal and not fabrication:
- Subtitle truncated ("Attention Is All You Need" vs the full title)
- Preprint vs published (arXiv vs conference, year off by 1)
- Name format ("J. Smith" / "John Smith" / "Smith, J.")
- Author list cut with "et al."
- Venue abbreviation ("NeurIPS" / "NIPS" / "Advances in Neural \
  Information Processing Systems")

These are suspicious:
- Title matches but completely different authors
- Year off by 2+ outside a preprint gap
- DOI resolves to a different paper
- Fields point to different papers (blended reference)

## How to investigate

Start from the comparison results. If the mismatch is obviously \
cosmetic, decide immediately — no tools needed.

When venue or year differs, search another database for the version \
the reference claims. A reference saying "Nature 2019" while the \
database found "arXiv 2020" might just mean the pipeline found the \
preprint — search CrossRef or OpenAlex for the published version.

When venue names look different but might be the same \
("Proceedings of JMLR" vs "Journal of Machine Learning Research"), \
search to confirm rather than guessing.

When in doubt, search. You have CrossRef, Semantic Scholar, OpenAlex, \
and PubMed available.

## Verdict

- **VALID** — cosmetic or explained. Same paper.
- **FABRICATED** — real mismatch. Wrong or non-existent paper.
- **UNVERIFIABLE** — not enough evidence to decide.

Report all discrepancies in field_discrepancies even when VALID."""


METADATA_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "metadata_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["VALID", "FABRICATED", "UNVERIFIABLE"],
                },
                "explanation": {
                    "type": "string",
                    "description": "Reasoning for the verdict.",
                },
                "flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concerns or notes for human review.",
                },
                "field_discrepancies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "ref_value": {"type": ["string", "null"]},
                            "db_value": {"type": ["string", "null"]},
                            "status": {"type": "string"},
                            "investigation_notes": {"type": ["string", "null"]},
                        },
                        "required": [
                            "field", "ref_value", "db_value", "status",
                            "investigation_notes",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "verdict", "explanation", "flags", "field_discrepancies",
            ],
            "additionalProperties": False,
        },
    },
}


_METADATA_TOOL_NAMES = {
    "search_crossref_by_doi",
    "search_semantic_scholar",
    "search_openalex",
    "search_pubmed",
    "compare_titles",
    "compare_authors",
    "verify_web_source",
}

METADATA_TOOL_DEFINITIONS = [
    t for t in TOOL_DEFINITIONS if t["function"]["name"] in _METADATA_TOOL_NAMES
]


def build_metadata_user_message(
    ref_title: Optional[str],
    ref_authors: list[str],
    ref_year: Optional[int],
    ref_venue: Optional[str],
    ref_doi: Optional[str],
    ref_raw_text: str,
    citation_format: Optional[str],
    db_title: Optional[str],
    db_authors: list[str],
    db_year: Optional[int],
    db_venue: Optional[str],
    db_doi: Optional[str],
    db_source: Optional[str],
    title_comparison: Optional[dict],
    author_comparison: Optional[dict],
    triage_reason: str,
) -> str:
    parts: list[str] = []

    parts.append("## Why This Reference Was Flagged")
    parts.append(f"**Triage reason**: {triage_reason}")

    parts.append("\n## Reference (from the paper's bibliography)")
    parts.append(f"- **Title**: {ref_title or '(not available)'}")
    if ref_authors:
        parts.append(f"- **Authors**: {', '.join(ref_authors)}")
    if ref_year:
        parts.append(f"- **Year**: {ref_year}")
    if ref_venue:
        parts.append(f"- **Venue**: {ref_venue}")
    if ref_doi:
        parts.append(f"- **DOI**: {ref_doi}")
    if citation_format:
        parts.append(f"- **Citation Format**: {citation_format}")
    parts.append(f"- **Raw text**: {ref_raw_text[:300]}")

    parts.append("\n## Database Record (best match)")
    parts.append(f"- **Source**: {db_source or 'unknown'}")
    parts.append(f"- **Title**: {db_title or '(not available)'}")
    if db_authors:
        parts.append(f"- **Authors**: {', '.join(db_authors)}")
    if db_year:
        parts.append(f"- **Year**: {db_year}")
    if db_venue:
        parts.append(f"- **Venue**: {db_venue}")
    if db_doi:
        parts.append(f"- **DOI**: {db_doi}")

    if title_comparison:
        parts.append("\n## Pre-Computed Title Comparison")
        parts.append(f"- **Exact match**: {title_comparison.get('exact_match', '?')}")
        parts.append(f"- **Similarity**: {title_comparison.get('similarity', '?')}")
        diffs = title_comparison.get("differences", [])
        if diffs:
            parts.append(f"- **Differences**: {'; '.join(diffs)}")
        else:
            parts.append("- **Differences**: none")

    if author_comparison:
        parts.append("\n## Pre-Computed Author Comparison")
        parts.append(f"- **Matched**: {author_comparison.get('matched_count', '?')}")
        parts.append(f"- **Unmatched**: {author_comparison.get('unmatched_count', '?')}")
        unmatched = author_comparison.get("unmatched_authors", [])
        if unmatched:
            parts.append(f"- **Unmatched authors**: {', '.join(unmatched)}")
        parts.append(f"- **Truncation expected**: {author_comparison.get('truncation_expected', '?')}")
        if author_comparison.get("explanation"):
            parts.append(f"- **Notes**: {author_comparison['explanation']}")

    parts.append("\n## Your Task")
    parts.append(
        "Investigate the flagged mismatch. Is it cosmetic or real? "
        "If the pre-computed comparison is sufficient to decide, do so "
        "immediately. Only search databases if you need to verify a "
        "specific hypothesis (e.g., checking a DOI, finding an alternate version)."
    )

    return "\n".join(parts)


class MetadataAgent:

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        tool_executor: ToolExecutor,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_tool_rounds: int = 3,
        max_tokens: int = 1024,
        timeout: int = 60,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self._client = openai_client
        self._tools = tool_executor
        self._model = model
        self._temperature = temperature
        self._max_tool_rounds = max_tool_rounds
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._cost_tracker = cost_tracker or CostTracker()

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    async def investigate(self, user_message: str) -> dict:
        messages: list[dict] = [
            {"role": "system", "content": METADATA_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        from citeextract.verification.api_clients.llm_client import (
            is_reasoning_model, effective_max_completion_tokens,
        )

        for round_num in range(self._max_tool_rounds):
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "max_completion_tokens": effective_max_completion_tokens(
                    self._model, self._max_tokens,
                ),
                "timeout": self._timeout,
                "tools": METADATA_TOOL_DEFINITIONS,
                "tool_choice": "auto",
            }
            if not is_reasoning_model(self._model):
                kwargs["temperature"] = self._temperature
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if response.usage:
                self._cost_tracker.add(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                break

            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    log.warning(f"MetadataAgent: invalid JSON args for {fn_name}, using empty dict")
                    fn_args = {}

                log.debug(f"MetadataAgent calling tool: {fn_name}({fn_args})")
                result = await self._tools.execute(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })
        else:
            log.warning(f"MetadataAgent hit max_tool_rounds ({self._max_tool_rounds})")

        return await self._get_verdict(messages)

    async def investigate_followup(self, followup_message: str) -> dict:
        gs_tools = [
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] == "search_google_scholar"
        ]

        messages: list[dict] = [
            {"role": "system", "content": METADATA_SYSTEM_PROMPT},
            {"role": "user", "content": followup_message},
        ]

        from citeextract.verification.api_clients.llm_client import (
            is_reasoning_model, effective_max_completion_tokens,
        )

        for _ in range(2):
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "max_completion_tokens": effective_max_completion_tokens(
                    self._model, self._max_tokens,
                ),
                "timeout": self._timeout,
                "tools": gs_tools,
                "tool_choice": "auto",
            }
            if not is_reasoning_model(self._model):
                kwargs["temperature"] = self._temperature
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if response.usage:
                self._cost_tracker.add(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                break

            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    log.warning(f"MetadataAgent followup: invalid JSON args for {fn_name}")
                    fn_args = {}

                log.debug(f"MetadataAgent followup calling: {fn_name}({fn_args})")
                result = await self._tools.execute(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })

        return await self._get_verdict(messages)

    async def _get_verdict(self, messages: list[dict]) -> dict:
        messages_copy = list(messages)
        messages_copy.append({
            "role": "user",
            "content": "Provide your final metadata verdict now as JSON.",
        })

        from citeextract.verification.api_clients.llm_client import (
            is_reasoning_model, effective_max_completion_tokens,
        )

        try:
            verdict_kwargs: dict = {
                "model": self._model,
                "messages": messages_copy,
                "max_completion_tokens": effective_max_completion_tokens(
                    self._model, self._max_tokens,
                ),
                "timeout": self._timeout,
                "response_format": METADATA_VERDICT_SCHEMA,
            }
            if not is_reasoning_model(self._model):
                verdict_kwargs["temperature"] = self._temperature
            response = await self._client.chat.completions.create(**verdict_kwargs)

            if response.usage:
                self._cost_tracker.add(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            raw = response.choices[0].message.content or ""
            return json.loads(raw)
        except Exception as e:
            log.error(f"MetadataAgent verdict failed: {e}")
            return {
                "verdict": "UNVERIFIABLE",
                "explanation": f"Metadata agent failed: {str(e)[:200]}",
                "flags": ["metadata_agent_error"],
                "field_discrepancies": [],
            }
