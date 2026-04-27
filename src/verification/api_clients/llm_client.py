"""LLM client — general-purpose interface for OpenAI API calls.

Used by:
  - Agentic verification (agent loop via separate OpenAI client)
  - Reference re-parsing (llm_ref_parser.py via raw_chat)
  - Passage-based claim verification (via raw_chat)
"""

import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel

from src import config
from src.verification import spend_guard

log = logging.getLogger(__name__)


# ---- Cost tracking ----

class CostTracker(BaseModel):
    """Tracks token usage and estimates cost."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
        # Mirror this call's marginal cost into the persistent monthly ledger
        # so spend_guard.check_budget() at request entry points sees a global
        # total across every code path that constructs a CostTracker.
        pricing = config.llm_pricing()
        marginal = (
            input_tokens * pricing["input_cost_per_million"] / 1_000_000
            + output_tokens * pricing["output_cost_per_million"] / 1_000_000
        )
        spend_guard.record(marginal)

    @property
    def estimated_cost_usd(self) -> float:
        pricing = config.llm_pricing()
        return (
            self.total_input_tokens * pricing["input_cost_per_million"] / 1_000_000
            + self.total_output_tokens * pricing["output_cost_per_million"] / 1_000_000
        )


# ---- Abstract interface ----

class LLMClient(ABC):
    """Base class for LLM API clients."""

    cost_tracker: CostTracker

    @abstractmethod
    async def raw_chat(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        """Send a raw system+user prompt pair and return the response text.

        Tracks cost automatically.
        """
        ...


# ---- OpenAI implementation ----

class OpenAIClient(LLMClient):
    """OpenAI GPT client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 30,
    ):
        from openai import AsyncOpenAI

        key = api_key or config.openai_api_key()
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var "
                "or pass api_key parameter."
            )
        self._client = AsyncOpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cost_tracker = CostTracker()

    async def raw_chat(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        """Send a raw prompt pair and return the response text."""
        import httpx
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        if response.usage:
            self.cost_tracker.add(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        return response.choices[0].message.content or ""


# ---- Factory ----

def create_llm_client(llm_cfg: dict) -> LLMClient:
    """Create an LLM client from config dict."""
    provider = llm_cfg.get("provider", "openai")
    if provider == "openai":
        return OpenAIClient(
            model=llm_cfg.get("model", "gpt-4o-mini"),
            api_key=llm_cfg.get("api_key") or config.openai_api_key(),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 1024),
            timeout=llm_cfg.get("timeout", 30),
        )
    raise ValueError(f"Unknown LLM provider: {provider}. Supported: openai")
