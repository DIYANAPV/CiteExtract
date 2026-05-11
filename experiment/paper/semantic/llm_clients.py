from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

log = logging.getLogger(__name__)


PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":      (0.15,  0.60),
    "gpt-4o":           (2.50, 10.00),
    "gpt-5":            (1.25, 10.00),
    "gpt-5.5":          (5.00, 30.00),
    "gemini-2.5-flash": (0.30,  2.50),
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

    def cost_usd(self, model: str) -> float:
        if self.prompt_tokens == 0 and self.completion_tokens == 0:
            return 0.0
        api_model, _ = resolve_model(model)
        in_rate, out_rate = PRICING[api_model]
        return (
            self.prompt_tokens * in_rate / 1_000_000
            + self.completion_tokens * out_rate / 1_000_000
        )


class LLMClient(ABC):
    model: str
    temperature: float
    timeout_s: float

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


class OpenAIBenchClient(LLMClient):
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


class GeminiBenchClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing")
        self._client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    async def call(
        self, system: str, user: str, max_tokens: int = 512
    ) -> LLMCallResult:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )

        async def _one_shot():
            return await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=cfg,
                ),
                timeout=self.timeout_s,
            )

        t0 = time.perf_counter()
        try:
            resp = await _call_with_retries(_one_shot)
        except asyncio.TimeoutError:
            return LLMCallResult("", 0, 0, time.perf_counter() - t0, error="timeout")
        except Exception as e:
            return LLMCallResult("", 0, 0, time.perf_counter() - t0,
                                 error=f"api_error:{type(e).__name__}:{str(e)[:120]}")

        latency = time.perf_counter() - t0
        text = (resp.text or "") if resp is not None else ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        return LLMCallResult(text, in_tok, out_tok, latency)


def build_client(model: str) -> LLMClient:
    if model.startswith("gpt-"):
        return OpenAIBenchClient(model=model)
    if model.startswith("gemini-"):
        return GeminiBenchClient(model=model)
    raise ValueError(f"Unknown model: {model}")
