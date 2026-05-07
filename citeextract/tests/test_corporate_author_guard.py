
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from citeextract.models.reference import Reference
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.existence import _cross_validate_authors
from citeextract.verification.matching import _author_tokens, _is_consortium_name


def _run(coro):
    return asyncio.run(coro)


class TestCorporateNameDetection:
    @pytest.mark.parametrize("name", [
        "Meta AI",
        "Google AI",
        "Google Research",
        "Google Brain",
        "OpenAI",
        "Anthropic",
        "DeepMind",
        "Google DeepMind",
        "Microsoft Research",
        "IBM Research",
        "Apple",
        "Amazon Science",
        "Cohere",
        "Mistral AI",
        "Stability AI",
        "Hugging Face",
        "Allen Institute for AI",
        "Qwen Team",
        "DeepSeek-AI",
    ])
    def test_known_corporate_names_detected(self, name):
        assert _is_consortium_name(name), f"{name!r} should be flagged corporate"

    @pytest.mark.parametrize("name", [
        "Yann LeCun",
        "Geoffrey Hinton",
        "Aakanksha Chowdhery",
        "Karl Cobbe",
        "Smith, J.",
        "Jane M. Doe",
    ])
    def test_individual_names_pass_through(self, name):
        assert not _is_consortium_name(name), f"{name!r} should not be flagged"

    @pytest.mark.parametrize("name", [
        "Some Org AI",
        "Tiny Lab",
        "Unknown Research",
    ])
    def test_org_suffix_pattern_catches_long_tail(self, name):
        assert _is_consortium_name(name)


class TestAuthorTokensFiltersCorporate:

    def test_meta_ai_filtered(self):
        assert _author_tokens(["Meta AI"]) == set()

    def test_individuals_kept(self):
        tokens = _author_tokens(["Yann LeCun", "Meta AI"])
        assert "lecun" in tokens
        assert "ai" not in tokens
        assert "meta" not in tokens


class TestCrossValidationSkipsCorporate:
    def test_org_only_ref_authors_skip_cross_val_with_flag(self):
        ref = Reference(
            ref_id="t1",
            title="The Llama 3 Herd of Models",
            authors=["Meta AI"],
            source_format="text",
        )
        result = ExistenceResult(
            ref_id="t1",
            status="FOUND",
            source="semantic_scholar",
            matched_title="The Llama 3 Herd of Models",
            matched_authors=["Aaron Grattafiori", "Abhimanyu Dubey"],
            databases_checked=["semantic_scholar"],
            flags=[],
        )
        client = AsyncMock()

        with patch(
            "citeextract.verification.api_clients.openalex.search_by_title",
            new=AsyncMock(side_effect=AssertionError("openalex must not be called")),
        ), patch(
            "citeextract.verification.api_clients.semantic_scholar.search_by_title",
            new=AsyncMock(side_effect=AssertionError("S2 must not be called")),
        ):
            _run(_cross_validate_authors(ref, result, client))

        joined = " | ".join(result.flags)
        assert "authors_corporate" in joined, (
            f"missing authors_corporate flag — got: {joined}"
        )
        assert "suspect_authors" not in joined, (
            "regression: corporate-only author list got the suspect-author flag"
        )

    def test_individual_authors_still_cross_validated(self):
        ref = Reference(
            ref_id="t2",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            source_format="text",
        )
        result = ExistenceResult(
            ref_id="t2",
            status="FOUND",
            source="semantic_scholar",
            matched_title="Attention Is All You Need",
            matched_authors=["Ashish Vaswani", "Noam Shazeer"],
            databases_checked=["semantic_scholar"],
            flags=[],
        )
        client = AsyncMock()
        with patch(
            "citeextract.verification.api_clients.openalex.search_by_title",
            new=AsyncMock(return_value={"authors": ["Ashish Vaswani"]}),
        ):
            _run(_cross_validate_authors(ref, result, client))

        joined = " | ".join(result.flags)
        assert "authors_cross_validated" in joined
