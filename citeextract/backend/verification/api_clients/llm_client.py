
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel

from citeextract import config
from citeextract.verification import spend_guard

log = logging.getLogger(__name__)


class CostTracker(BaseModel):

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
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


class LLMClient(ABC):

    cost_tracker: CostTracker

    @abstractmethod
    async def raw_chat(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        ...


_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

_REASONING_MIN_COMPLETION_TOKENS = 8192


def is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_MODEL_PREFIXES)


def effective_max_completion_tokens(model: str, configured: int) -> int:
    if is_reasoning_model(model):
        return max(configured, _REASONING_MIN_COMPLETION_TOKENS)
    return configured


class OpenAIClient(LLMClient):

    def __init__(
        self,
        model: str = "gpt-5-mini",
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
        import httpx
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": effective_max_completion_tokens(self.model, max_tokens),
            "response_format": {"type": "json_object"},
            "timeout": timeout,
        }
        if not is_reasoning_model(self.model):
            kwargs["temperature"] = self.temperature
        response = await self._client.chat.completions.create(**kwargs)
        if response.usage:
            self.cost_tracker.add(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        return response.choices[0].message.content or ""


def create_llm_client(llm_cfg: dict) -> LLMClient:
    provider = llm_cfg.get("provider", "openai")
    if provider == "openai":
        return OpenAIClient(
            model=llm_cfg.get("model", "gpt-5-mini"),
            api_key=llm_cfg.get("api_key") or config.openai_api_key(),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 1024),
            timeout=llm_cfg.get("timeout", 30),
        )
    raise ValueError(f"Unknown LLM provider: {provider}. Supported: openai")
