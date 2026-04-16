"""Claim Agent — focused LLM agent for verifying citing sentence accuracy.

Only invoked when the triage step identifies substantive citing claims.
Receives pre-retrieved passages from the cited paper and judges whether
the citing sentences accurately represent the source.

Key improvement over the monolithic CitationAgent: **context-awareness**.
The prompt adapts based on:
- How many context sentences are available (1 vs 3-4)
- Whether full text or only abstract was available
- Whether passages were pre-retrieved or need retrieval via tool
"""

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from src.models.comprehension import ClaimVerdict
from src.verification.api_clients.llm_client import CostTracker
from src.verification.agentic.tools import TOOL_DEFINITIONS, ToolExecutor
from src.verification.triage import ContextQuality

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt (claim verification only)
# ---------------------------------------------------------------------------

CLAIM_SYSTEM_PROMPT = """\
Check whether a citing sentence accurately represents the cited paper.

You receive the citation marker (e.g. `[15]`, `(Smith et al., 2020)`), \
the cited paper's title, passages from it, and the citing sentence with \
surrounding context.

## First: figure out what THIS citation actually claims

Sentences often cite multiple papers. Look at where the marker sits in \
the sentence — it tells you what is attributed to this specific paper.

"We used X (A et al.) and Y (B et al.) on dataset Z (C et al.)"
→ A et al. is cited for X. B et al. for Y. C et al. for Z.
   Do not verify the whole sentence against one paper.

Common citation purposes:
- Model/tool/dataset usage ("we used X [1]") — the paper just needs to \
  describe X. The experiment is the current authors' work.
- Method reference ("following [5]") — the paper needs to describe the \
  method. If the sentence is vague about what was followed, pull more \
  context from the current paper before judging.
- Factual claim ("X showed 30% improvement [2]") — check the number.
- Background ("several approaches exist [3,4]") — paper just needs to \
  be relevant.

## Then: decide if you need more information

If the citing sentence is vague about what was actually done — "we \
followed [5]", "we adapted [3]", "similar to [8]" — you may not have \
enough context to judge. Use retrieve_current_paper_context to search \
the current paper for what the authors actually did.

Skip this step when the claim is self-contained — "we used BERT [1]" \
or "accuracy was 95% [2]" need no extra context.

## Finally: verdict

- **SUPPORTS** — the sub-claim checks out against the passages. For \
  model/tool citations, confirming the paper describes that model/tool \
  is sufficient.
- **CONTRADICTS** — the passages say something different from what the \
  citing sentence attributes to this paper, or the sentence exaggerates \
  or cherry-picks.
- **NEUTRAL** — the passages do not clearly address the sub-claim. \
  Normal for background citations.

Ground your judgment in the passages and abstract only. Quote the \
relevant text. Evaluate each citing context separately.

## Tools
- **retrieve_cited_paper_passages** — re-retrieve from the cited paper \
  if pre-retrieved passages miss the sub-claim entirely.
- **retrieve_current_paper_context** — search the current paper \
  (the one containing the citation) for more detail about what the \
  authors actually did. Use when the citing sentence alone is too vague."""


# ---------------------------------------------------------------------------
# Claim verdict JSON schema
# ---------------------------------------------------------------------------

CLAIM_VERDICT_SCHEMA: dict[str, Any] = {
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
                                "description": "Type of citation: model_usage, method_reference, factual_claim, theoretical, background, contrast.",
                            },
                            "sub_claim": {
                                "type": "string",
                                "description": "The specific sub-claim attributed to this reference (not the full sentence).",
                            },
                            "verdict": {
                                "type": "string",
                                "enum": ["SUPPORTS", "CONTRADICTS", "NEUTRAL"],
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
            "NOTE: Only the citing sentence itself was available (it sits at "
            "a paragraph boundary). No surrounding sentences could be "
            "extracted. Be conservative — lean toward NEUTRAL when evidence "
            "is ambiguous."
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
            "\n**IMPORTANT**: Only the paper's abstract was available "
            "(full text is paywalled or unavailable). Specific method/result "
            "claims cannot be fully verified against the abstract alone. "
            "Lean toward NEUTRAL for detailed claims."
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
            {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # --- Tool-calling loop (usually 0 rounds — just judgment) ---
        for round_num in range(self._max_tool_rounds):
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
                    log.warning(f"ClaimAgent: invalid JSON args for {fn_name}, using empty dict")
                    fn_args = {}

                log.debug(f"ClaimAgent calling tool: {fn_name}({fn_args})")
                result = await self._tools.execute(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })
        else:
            log.warning(f"ClaimAgent hit max_tool_rounds ({self._max_tool_rounds})")

        # --- Final structured verdict ---
        return await self._get_verdicts(messages)

    async def _get_verdicts(self, messages: list[dict]) -> list[ClaimVerdict]:
        """Get structured claim verdicts."""
        messages_copy = list(messages)
        messages_copy.append({
            "role": "user",
            "content": "Provide your final claim verdicts now as JSON.",
        })

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages_copy,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                response_format=CLAIM_VERDICT_SCHEMA,
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
                    verdict=v.get("verdict", "NEUTRAL"),
                    explanation=v.get("explanation", ""),
                    evidence_quote=v.get("evidence_quote", ""),
                ))

            # Guard: LLM returned fewer verdicts than expected
            if not verdicts:
                log.warning("ClaimAgent returned empty verdicts array")
                verdicts = [
                    ClaimVerdict(verdict="NEUTRAL", explanation="No verdict returned by agent.")
                    for _ in range(max(1, self._expected_count))
                ]
            return verdicts

        except Exception as e:
            log.error(f"ClaimAgent verdict failed: {e}")
            # Return one NEUTRAL verdict per expected citing context
            return [
                ClaimVerdict(
                    verdict="NEUTRAL",
                    explanation=f"Claim agent failed: {str(e)[:200]}",
                    evidence_quote="",
                )
                for _ in range(max(1, self._expected_count))
            ]
