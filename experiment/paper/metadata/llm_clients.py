from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o":      (2.50, 10.00),
    "gpt-5":       (1.25, 10.00),
    "gpt-5.5":     (5.00, 30.00),
}

WEB_SEARCH_COST_PER_CALL: dict[str, float] = {
    "gpt-4o-mini": 0.025,
    "gpt-4o":      0.025,
    "gpt-5":       0.010,
    "gpt-5.5":     0.010,
}

_EFFORT_SUFFIXES = ("-min", "-med")
_EFFORT_BY_MODEL: dict[str, dict[str, str]] = {
    "gpt-5":   {"-min": "minimal", "-med": "medium"},
    "gpt-5.5": {"-min": "low",     "-med": "medium"},
}

_REASONING_TOKEN_BUDGET: dict[str, int] = {
    "minimal": 2048,
    "low":     4096,
    "medium":  8192,
    "high":    16384,
}


def resolve_model(logical: str) -> tuple[str, str | None]:
    for suf in _EFFORT_SUFFIXES:
        if logical.endswith(suf):
            base = logical[: -len(suf)]
            mapping = _EFFORT_BY_MODEL.get(base)
            if mapping is not None:
                return base, mapping[suf]
    return logical, None


def _effective_cap(max_tokens: int, effort: str | None) -> int:
    if effort is None:
        return max_tokens
    return max(max_tokens, _REASONING_TOKEN_BUDGET.get(effort, 8192))


@dataclass
class LLMCallResult:
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    error: str | None = None
    n_search_invocations: int = 0
    search_queries: list[str] = field(default_factory=list)

    def cost_usd(self, model: str) -> float:
        if self.prompt_tokens == 0 and self.completion_tokens == 0 and self.n_search_invocations == 0:
            return 0.0
        api_model, _ = resolve_model(model)
        in_rate, out_rate = PRICING[api_model]
        token_cost = (
            self.prompt_tokens * in_rate / 1_000_000
            + self.completion_tokens * out_rate / 1_000_000
        )
        search_rate = WEB_SEARCH_COST_PER_CALL.get(api_model, 0.025)
        return token_cost + self.n_search_invocations * search_rate


class LLMClient(ABC):
    model: str

    @abstractmethod
    async def call(
        self, system: str, user: str, max_tokens: int = 512
    ) -> LLMCallResult:
        ...


async def _call_with_retries(
    fn,
    *,
    max_retries: int = 5,
    base_delay: float = 2.0,
    factor: float = 2.0,
    jitter: float = 0.5,
):
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            msg = str(e).lower()
            transient = (
                "rate" in msg or "429" in msg
                or "timeout" in msg or "503" in msg or "502" in msg or "500" in msg
                or "overloaded" in msg or "unavailable" in msg or "deadline" in msg
            )
            last_exc = e
            if not transient or attempt == max_retries:
                raise
            delay = base_delay * (factor ** attempt) + random.uniform(0, jitter)
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc


class OpenAIChatBenchClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ):
        from openai import AsyncOpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing")
        self._client = AsyncOpenAI(api_key=key)
        self.model = model
        self.api_model, self.reasoning_effort = resolve_model(model)
        self.temperature = temperature
        self.timeout_s = timeout_s

    async def call(
        self, system: str, user: str, max_tokens: int = 512
    ) -> LLMCallResult:
        import httpx
        from openai import APITimeoutError

        timeout = httpx.Timeout(connect=15.0, read=self.timeout_s, write=15.0, pool=15.0)

        async def _one_shot():
            kwargs: dict = dict(
                model=self.api_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                timeout=timeout,
            )
            if self.reasoning_effort is not None:
                kwargs["max_completion_tokens"] = _effective_cap(max_tokens, self.reasoning_effort)
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
                kwargs["max_tokens"] = max_tokens
            return await self._client.chat.completions.create(**kwargs)

        t0 = time.perf_counter()
        try:
            resp = await _call_with_retries(_one_shot)
        except APITimeoutError:
            return LLMCallResult("", 0, 0, time.perf_counter() - t0, error="timeout")
        except Exception as e:
            return LLMCallResult("", 0, 0, time.perf_counter() - t0,
                                 error=f"api_error:{type(e).__name__}:{str(e)[:120]}")

        latency = time.perf_counter() - t0
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        return LLMCallResult(text, in_tok, out_tok, latency)


class OpenAIResponsesBenchClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 90.0,
    ):
        from openai import AsyncOpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing")
        self._client = AsyncOpenAI(api_key=key)
        self.model = model
        self.api_model, self.reasoning_effort = resolve_model(model)
        self.temperature = temperature
        self.timeout_s = timeout_s

    async def call(
        self, system: str, user: str, max_tokens: int = 512
    ) -> LLMCallResult:
        import httpx
        from openai import APITimeoutError

        timeout = httpx.Timeout(connect=15.0, read=self.timeout_s, write=15.0, pool=15.0)

        effort = self.reasoning_effort
        if effort == "minimal":
            effort = "low"

        async def _one_shot():
            kwargs: dict = dict(
                model=self.api_model,
                tools=[{"type": "web_search_preview"}],
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=_effective_cap(max_tokens, effort),
                timeout=timeout,
            )
            if effort is not None:
                kwargs["reasoning"] = {"effort": effort}
            else:
                kwargs["temperature"] = self.temperature
            return await self._client.responses.create(**kwargs)

        t0 = time.perf_counter()
        try:
            resp = await _call_with_retries(_one_shot)
        except APITimeoutError:
            return LLMCallResult("", 0, 0, time.perf_counter() - t0, error="timeout")
        except Exception as e:
            return LLMCallResult("", 0, 0, time.perf_counter() - t0,
                                 error=f"api_error:{type(e).__name__}:{str(e)[:120]}")

        latency = time.perf_counter() - t0
        text = (resp.output_text or "") if resp is not None else ""

        n_search = 0
        queries: list[str] = []
        for item in (resp.output or []):
            if getattr(item, "type", None) == "web_search_call":
                n_search += 1
                action = getattr(item, "action", None)
                q = getattr(action, "query", None) if action is not None else None
                if q:
                    queries.append(q)

        in_tok = getattr(resp.usage, "input_tokens", 0) or 0
        out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        return LLMCallResult(
            text, in_tok, out_tok, latency,
            n_search_invocations=n_search, search_queries=queries,
        )


def build_client(model: str, condition: str) -> LLMClient:
    if condition == "llm_only":
        return OpenAIChatBenchClient(model=model)
    if condition == "llm_with_search":
        return OpenAIResponsesBenchClient(model=model)
    raise ValueError(f"Unknown condition: {condition}")
