
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from citeextract.models.comprehension import FullTextResult
from citeextract.models.verdict import ExistenceResult
from citeextract.verification.api_clients.fulltext import (
    get_full_text,
    _unpaywall_pdf_url,
    _get_arxiv_id,
    _extract_from_pdf,
)


def _run(coro):
    return asyncio.run(coro)


def _make_existence(
    ref_id: str = "1",
    status: str = "FOUND",
    doi: str = "10.1234/test",
    oa_url: str = None,
    abstract: str = "This is the abstract of the paper.",
) -> ExistenceResult:
    return ExistenceResult(
        ref_id=ref_id,
        status=status,
        source="semantic_scholar",
        matched_title="Test Paper Title",
        matched_authors=["Smith", "Jones"],
        matched_year=2023,
        matched_doi=doi,
        abstract=abstract,
        oa_url=oa_url,
        databases_checked=["semantic_scholar"],
    )


def _make_cache():
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


class TestGetArxivId:
    def test_from_arxiv_doi(self):
        er = _make_existence(doi="10.48550/arxiv.2305.14314")
        assert _get_arxiv_id(er) == "2305.14314"

    def test_from_oa_url(self):
        er = _make_existence(oa_url="https://arxiv.org/pdf/2305.14314v1")
        assert _get_arxiv_id(er) == "2305.14314v1"

    def test_from_abs_url(self):
        er = _make_existence(oa_url="https://arxiv.org/abs/2305.14314")
        assert _get_arxiv_id(er) == "2305.14314"

    def test_no_arxiv(self):
        er = _make_existence(doi="10.1038/nature12373", oa_url=None)
        assert _get_arxiv_id(er) is None

    def test_non_arxiv_oa_url(self):
        er = _make_existence(oa_url="https://www.nature.com/articles/nature12373.pdf")
        assert _get_arxiv_id(er) is None


class TestUnpaywallPdfUrl:
    def test_returns_best_oa_pdf(self):
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "best_oa_location": {
                "url_for_pdf": "https://example.com/paper.pdf",
            },
            "oa_locations": [],
        }

        with patch(
            "citeextract.verification.api_clients.fulltext.fetch_with_retry",
            return_value=mock_resp,
        ):
            url = _run(_unpaywall_pdf_url("10.1234/test", mock_client))
            assert url == "https://example.com/paper.pdf"

    def test_falls_back_to_oa_locations(self):
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "best_oa_location": {"url_for_pdf": None},
            "oa_locations": [
                {"url_for_pdf": None},
                {"url_for_pdf": "https://repo.edu/paper.pdf"},
            ],
        }

        with patch(
            "citeextract.verification.api_clients.fulltext.fetch_with_retry",
            return_value=mock_resp,
        ):
            url = _run(_unpaywall_pdf_url("10.1234/test", mock_client))
            assert url == "https://repo.edu/paper.pdf"

    def test_returns_none_on_404(self):
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch(
            "citeextract.verification.api_clients.fulltext.fetch_with_retry",
            return_value=mock_resp,
        ):
            url = _run(_unpaywall_pdf_url("10.1234/test", mock_client))
            assert url is None


class TestFullTextCache:
    def test_returns_cached_result(self):
        cache = _make_cache()
        cached_data = {
            "source": "s2_api",
            "full_text": "Cached full text of the paper.",
            "sections": [{"name": "Intro", "text": "Cached intro text."}],
            "abstract": "Cached abstract.",
        }
        cache.get = AsyncMock(return_value=cached_data)

        er = _make_existence()
        mock_client = AsyncMock()

        result = _run(get_full_text(er, mock_client, cache))
        assert result.source == "s2_api"
        assert result.full_text == "Cached full text of the paper."

    def test_abstract_fallback_when_not_found(self):
        cache = _make_cache()
        er = _make_existence(status="NOT_FOUND", abstract="The abstract text.")
        mock_client = AsyncMock()

        result = _run(get_full_text(er, mock_client, cache))
        assert result.source == "not_found"
        assert result.abstract == "The abstract text."

    def test_abstract_fallback_when_no_pdf_available(self):
        cache = _make_cache()
        er = _make_existence(doi=None, oa_url=None, abstract="Fallback abstract.")
        mock_client = AsyncMock()

        with patch(
            "citeextract.verification.api_clients.fulltext._try_s2_fallback_pdf",
            new=AsyncMock(return_value=None),
        ), patch(
            "citeextract.verification.api_clients.fulltext._try_arxiv_fallback_pdf",
            new=AsyncMock(return_value=None),
        ):
            result = _run(get_full_text(er, mock_client, cache))
        assert result.source == "abstract_only"
        assert result.abstract == "Fallback abstract."


class TestPdfExtraction:
    def test_returns_not_found_when_grobid_unavailable(self):
        from citeextract.verification.api_clients import fulltext as ft_module
        ft_module._reset_grobid_state_for_tests()
        with patch(
            "citeextract.verification.api_clients.fulltext.requests.get",
            side_effect=ConnectionError("GROBID offline for this test"),
        ):
            result = _run(_extract_from_pdf("/nonexistent/path.pdf", source="user_pdf"))
        ft_module._reset_grobid_state_for_tests()
        assert result.source == "not_found"
        assert result.full_text is None


class TestGrobidConcurrency:
    def test_extract_runs_in_parallel_up_to_semaphore(self):
        from citeextract.verification.api_clients import fulltext as ft_module
        import time

        ft_module._reset_grobid_state_for_tests()

        sleep_s = 0.3
        n_calls = 8
        cap = 4

        ft_module._grobid_available = True
        ft_module._grobid_extract_semaphore = asyncio.Semaphore(cap)

        def _fake_extract(_pdf_path):
            time.sleep(sleep_s)
            return ("body text", [{"name": "S", "text": "body text"}])

        with patch(
            "citeextract.verification.api_clients.fulltext._extract_via_grobid",
            side_effect=_fake_extract,
        ):
            async def _all():
                return await asyncio.gather(*[
                    _extract_from_pdf(f"/tmp/fake_{i}.pdf", source="user_pdf")
                    for i in range(n_calls)
                ])

            t0 = time.perf_counter()
            results = _run(_all())
            elapsed = time.perf_counter() - t0

        ft_module._reset_grobid_state_for_tests()

        assert len(results) == n_calls
        assert all(r.source == "user_pdf" and r.full_text for r in results)

        min_expected = (n_calls / cap) * sleep_s * 0.7
        max_expected = (n_calls / cap) * sleep_s * 2.0
        assert min_expected <= elapsed <= max_expected, (
            f"elapsed={elapsed:.3f}s outside expected range "
            f"[{min_expected:.3f}, {max_expected:.3f}] for cap={cap}"
        )

    def test_availability_probe_runs_once_under_concurrency(self):
        from citeextract.verification.api_clients import fulltext as ft_module

        ft_module._reset_grobid_state_for_tests()

        call_count = 0

        def _fake_health(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.status_code = 200
            return response

        with patch(
            "citeextract.verification.api_clients.fulltext.requests.get",
            side_effect=_fake_health,
        ):
            async def _race():
                return await asyncio.gather(*[
                    ft_module._ensure_grobid_available() for _ in range(10)
                ])

            results = _run(_race())

        ft_module._reset_grobid_state_for_tests()

        assert all(results), "all callers should see GROBID as available"
        assert call_count == 1, (
            f"health probe ran {call_count} times — lock is not serializing first-callers"
        )


class TestPerRefFetchBudget:

    def test_timeout_returns_abstract_only_with_budget_attempt(self):
        from citeextract.verification.api_clients import fulltext as ft

        async def _slow_waterfall(*args, **kwargs):
            await asyncio.sleep(5)
            raise AssertionError("waterfall should have been cancelled")

        er = _make_existence(abstract="Abstract text of the paper.")

        with patch.object(ft, "_run_waterfall", side_effect=_slow_waterfall), \
             patch("citeextract.config.comprehension",
                   return_value={"per_ref_fetch_timeout_s": 0.05}):
            client = MagicMock()
            cache = _make_cache()
            result = _run(get_full_text(er, client, cache))

        assert result.source == "abstract_only"
        assert result.abstract == "Abstract text of the paper."
        assert any(
            a.source == "fetch_budget" and a.status == "timeout"
            for a in (result.attempts or [])
        ), "expected a fetch_budget timeout attempt entry on the result"

    def test_fast_path_returns_underlying_result_unchanged(self):
        from citeextract.verification.api_clients import fulltext as ft

        expected = FullTextResult(
            source="oa_url",
            full_text="The full body text.",
            abstract="abstract",
        )

        async def _quick_waterfall(*args, **kwargs):
            return expected

        er = _make_existence()

        with patch.object(ft, "_run_waterfall", side_effect=_quick_waterfall):
            client = MagicMock()
            cache = _make_cache()
            result = _run(get_full_text(er, client, cache))

        assert result is expected
