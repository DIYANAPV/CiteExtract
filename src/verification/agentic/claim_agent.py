"""Claim Agent — LLM agent for verifying citing sentence accuracy.

Invoked when the triage step identifies substantive citing claims.
Receives pre-retrieved passages from the cited paper and judges whether
the citing sentences accurately represent the source.

The prompt adapts based on:
- How many context sentences are available (1 vs 3-4)
- Whether full text or only abstract was available
- Whether passages were pre-retrieved or need retrieval via tool

The verdict scheme is selectable via `verdict_classes` (2 or 3). Prompt
files live in `prompts/claim_{2,3}class.txt` and the JSON schema for
structured output is built to match.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI

from src.models.comprehension import ClaimVerdict
from src.verification.api_clients.llm_client import CostTracker
from src.verification.agentic.tools import TOOL_DEFINITIONS, ToolExecutor
from src.verification.triage import ContextQuality

log = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Valid verdict values per scheme.
_VERDICT_VALUES: dict[int, list[str]] = {
    3: ["SUPPORTS", "CONTRADICTS", "NEUTRAL"],
    2: ["SUPPORTED", "NOT_SUPPORTED"],
}

# Default fallback verdict when the model can't decide or an error occurs.
_DEFAULT_VERDICT: dict[int, str] = {
    3: "NEUTRAL",
    2: "NOT_SUPPORTED",
}


def load_claim_prompt(verdict_classes: int) -> str:
    """Load the system prompt text for the chosen verdict scheme."""
    if verdict_classes not in _VERDICT_VALUES:
        raise ValueError(f"verdict_classes must be 2 or 3, got {verdict_classes}")
    path = _PROMPTS_DIR / f"claim_{verdict_classes}class.txt"
    return path.read_text(encoding="utf-8")


def build_verdict_schema(verdict_classes: int) -> dict[str, Any]:
    """Build the structured-output JSON schema for the chosen verdict scheme."""
    if verdict_classes not in _VERDICT_VALUES:
        raise ValueError(f"verdict_classes must be 2 or 3, got {verdict_classes}")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "claim_verdicts",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "citing_sentence": {
                                    "type": "string",
                                    "description": "The citing sentence being evaluated.",
                                },
                                "citation_type": {
                                    "type": "string",
                                    "enum": [
                                        "model_usage", "method_reference",
                                        "factual_claim", "theoretical",
                                        "background", "contrast", "other",
                                    ],
                                    "description": "Type of citation being made.",
                                },
                                "sub_claim": {
                                    "type": "string",
                                    "description": "The specific sub-claim attributed to this reference (not the full sentence).",
                                },
                                "verdict": {
                                    "type": "string",
                                    "enum": _VERDICT_VALUES[verdict_classes],
                                },
                                "explanation": {
                                    "type": "string",
                                    "description": "Reasoning for this verdict.",
                                },
                                "evidence_quote": {
                                    "type": "string",
                                    "description": "Most relevant quote from passages.",
                                },
                            },
                            "required": [
                                "citing_sentence", "citation_type", "sub_claim",
                                "verdict", "explanation", "evidence_quote",
                            ],
                            "additionalProperties": False,
                        },
                        "description": "One verdict per citing sentence.",
                    },
                },
                "required": ["verdicts"],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool subset for claim agent
# ---------------------------------------------------------------------------

_CLAIM_TOOL_NAMES = {"retrieve_cited_paper_passages", "retrieve_current_paper_context"}

CLAIM_TOOL_DEFINITIONS = [
    t for t in TOOL_DEFINITIONS if t["function"]["name"] in _CLAIM_TOOL_NAMES
]


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _format_context_quality_note(quality: ContextQuality) -> str:
    """Generate a context quality note for the prompt."""
    notes = []
    if quality.total_sentences == 1:
        notes.append(
            "NOTE: Only the citing sentence itself was available (no "
            "surrounding sentences). Focus on what the sentence explicitly "
            "claims about this specific paper."
        )
    else:
        if not quality.has_before:
            notes.append(
                "NOTE: This citation appears at the start of a paragraph. "
                "No preceding sentences are available for context."
            )
        if not quality.has_after:
            notes.append(
                "NOTE: This citation appears at the end of a paragraph. "
                "No following sentences are available."
            )
    return "\n".join(notes)


def build_claim_user_message(
    citing_contexts: list[dict],
    paper_title: str,
    paper_abstract: Optional[str],
    full_text_available: bool,
    full_text_source: Optional[str],
    paper_doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
) -> str:
    """Build the user message for the claim agent.

    Args:
        citing_contexts: List of dicts with keys:
            - citing_sentence (str)
            - context_before (str)
            - context_after (str)
            - context_quality (ContextQuality)
            - passages (list[dict] with text, section, score)
            - marker (str): citation marker, e.g. '[1]', '(Smith, 2020)'
        paper_title: Title of the cited paper.
        paper_abstract: Abstract of the cited paper (if available).
        full_text_available: Whether full text (not just abstract) was used.
        full_text_source: Source of the text (s2_api, arxiv, abstract_only, etc.).
        paper_doi: DOI for potential re-retrieval.
        arxiv_id: arXiv ID for potential re-retrieval.
    """
    parts: list[str] = []

    parts.append("## Cited Paper")
    parts.append(f"- **Title**: {paper_title}")
    if paper_doi:
        parts.append(f"- **DOI**: {paper_doi}")
    if arxiv_id:
        parts.append(f"- **arXiv ID**: {arxiv_id}")

    # Source quality note
    if full_text_source == "abstract_only" or not full_text_available:
        parts.append(
            "\nNote: Only the paper's abstract was available "
            "(full text is paywalled or unavailable). Base your "
            "judgment on the abstract and any retrieved passages."
        )
    else:
        parts.append(f"\n*Full text available via: {full_text_source}*")

    if paper_abstract:
        parts.append(f"\n### Abstract\n{paper_abstract[:1500]}")

    # Citing contexts with passages
    for i, ctx in enumerate(citing_contexts, 1):
        parts.append(f"\n---\n## Citing Context {i}")

        # Identify which citation in the sentence we are checking
        marker = ctx.get("marker", "")
        if marker:
            parts.append(f"**Citation marker being checked**: `{marker}`")
        parts.append(f"**Paper being checked**: {paper_title}")

        # Context quality note
        quality = ctx.get("context_quality")
        if quality:
            note = _format_context_quality_note(quality)
            if note:
                parts.append(note)

        # Show the citing sentence with context
        context_parts = []
        if ctx.get("context_before"):
            context_parts.append(ctx["context_before"].strip())
        context_parts.append(f"**{ctx['citing_sentence'].strip()}**")
        if ctx.get("context_after"):
            context_parts.append(ctx["context_after"].strip())
        parts.append("\n### Citing Sentence (bold) with Context")
        parts.append(" ".join(context_parts))

        # Pre-retrieved passages
        passages = ctx.get("passages", [])
        if passages:
            parts.append(f"\n### Retrieved Passages ({len(passages)} found)")
            for j, p in enumerate(passages, 1):
                section = f" (Section: {p.get('section', 'unknown')})" if p.get("section") else ""
                score = f" [score: {p.get('score', 0):.3f}]" if p.get("score") else ""
                parts.append(f"\n**Passage {j}**{section}{score}")
                parts.append(p.get("text", ""))
        else:
            parts.append(
                "\n### Retrieved Passages\n"
                "No passages were pre-retrieved. Use retrieve_cited_paper_passages "
                "to fetch relevant passages if needed."
            )

    parts.append("\n---\n## Your Task")
    parts.append(
        f"Evaluate each of the {len(citing_contexts)} citing sentence(s) above. "
        "For each, determine whether the passages support, contradict, or "
        "don't address the claim. Return one verdict per citing sentence."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Claim Agent class
# ---------------------------------------------------------------------------

class ClaimAgent:
    """Focused agent for verifying citing sentence accuracy."""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        tool_executor: ToolExecutor,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tool_rounds: int = 2,
        max_tokens: int = 1024,
        timeout: int = 60,
        cost_tracker: Optional[CostTracker] = None,
        verdict_classes: int = 3,
    ) -> None:
        self._client = openai_client
        self._tools = tool_executor
        self._model = model
        self._temperature = temperature
        self._max_tool_rounds = max_tool_rounds
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._cost_tracker = cost_tracker or CostTracker()
        self._verdict_classes = verdict_classes
        self._system_prompt = load_claim_prompt(verdict_classes)
        self._verdict_schema = build_verdict_schema(verdict_classes)
        self._default_verdict = _DEFAULT_VERDICT[verdict_classes]

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    async def verify_claims(
        self,
        user_message: str,
        expected_count: int = 1,
    ) -> list[ClaimVerdict]:
        """Run the claim verification and return verdicts per citing sentence.

        Args:
            user_message: The formatted user prompt with citing contexts.
            expected_count: Number of citing contexts (for correct fallback count).

        Returns:
            List of ClaimVerdict, one per citing sentence.
        """
        self._expected_count = expected_count
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]

        # --- Tool-calling loop (usually 0 rounds — just judgment) ---
        for _round in range(self._max_tool_rounds):
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout,
            }
            # Only offer tools if passages might need re-retrieval
            if CLAIM_TOOL_DEFINITIONS:
                kwargs["tools"] = CLAIM_TOOL_DEFINITIONS
                kwargs["tool_choice"] = "auto"

            response = await self._client.chat.completions.create(**kwargs)

            # #9: Guard against empty choices
            if not response.choices:
                log.error("ClaimAgent: API returned empty choices")
                break

            choice = response.choices[0]

            if response.usage:
                self._cost_tracker.add(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            # #8: Log when response was truncated
            if choice.finish_reason == "length":
                log.warning(
                    f"ClaimAgent: response truncated (finish_reason=length, "
                    f"max_tokens={self._max_tokens})"
                )

            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                break

            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    # #7: Skip the tool call instead of sending empty args
                    log.warning(f"ClaimAgent: invalid JSON args for {fn_name}, skipping tool call")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": f"Invalid arguments for {fn_name}"}),
                    })
                    continue

                log.debug(f"ClaimAgent calling tool: {fn_name}({fn_args})")
                result = await self._tools.execute(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })
        else:
            # Only a real concern when we actually allowed tool rounds and
            # exhausted them. With max_tool_rounds=0 this branch fires on
            # the trivial "0 iterations" case, which isn't noteworthy.
            if self._max_tool_rounds > 0:
                log.warning(f"ClaimAgent hit max_tool_rounds ({self._max_tool_rounds})")

        # --- Final structured verdict ---
        return await self._get_verdicts(messages)

    async def _get_verdicts(self, messages: list[dict]) -> list[ClaimVerdict]:
        """Get structured claim verdicts."""
        messages_copy = list(messages)
        messages_copy.append({
            "role": "user",
            "content": "Based on the evidence you reviewed above, provide your final verdicts as JSON. Include the most relevant quote for each verdict.",
        })

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages_copy,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                response_format=self._verdict_schema,
            )

            if response.usage:
                self._cost_tracker.add(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            raw = response.choices[0].message.content or ""
            data = json.loads(raw)

            verdicts = []
            for v in data.get("verdicts", []):
                verdicts.append(ClaimVerdict(
                    verdict=v.get("verdict", self._default_verdict),
                    explanation=v.get("explanation", ""),
                    evidence_quote=v.get("evidence_quote", ""),
                ))

            if not verdicts:
                log.error("ClaimAgent returned empty verdicts array")
                raise RuntimeError("ClaimAgent returned no verdicts")

            return verdicts

        except Exception as e:
            log.error(f"ClaimAgent verdict failed: {e}")
            raise
