"""Tests for full-text retrieval client — unit tests with mocked HTTP."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.comprehension import FullTextResult
from src.models.verdict import ExistenceResult
from src.verification.api_clients.fulltext import (
    get_full_text,
    _unpaywall_pdf_url,
    _get_arxiv_id,
    _extract_from_pdf,
)


def _run(coro):
    """Run an async coroutine synchronously (matches existing test pattern)."""
    return asyncio.run(coro)


# ---- Fixtures ----

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
    """Create a mock APICache."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)  # No cache hit by default
    cache.set = AsyncMock()
    return cache


# ---- arXiv ID extraction ----

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


# ---- Unpaywall ----

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
            "src.verification.api_clients.fulltext.fetch_with_retry",
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
            "src.verification.api_clients.fulltext.fetch_with_retry",
            return_value=mock_resp,
        ):
            url = _run(_unpaywall_pdf_url("10.1234/test", mock_client))
            assert url == "https://repo.edu/paper.pdf"

    def test_returns_none_on_404(self):
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch(
            "src.verification.api_clients.fulltext.fetch_with_retry",
            return_value=mock_resp,
        ):
            url = _run(_unpaywall_pdf_url("10.1234/test", mock_client))
            assert url is None


# ---- Cache behavior ----

class TestFullTextCache:
    def test_returns_cached_result(self):
        """Should return cached FullTextResult without making any API calls."""
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
        """When paper not found in L2, should return not_found with abstract."""
        cache = _make_cache()
        er = _make_existence(status="NOT_FOUND", abstract="The abstract text.")
        mock_client = AsyncMock()

        result = _run(get_full_text(er, mock_client, cache))
        assert result.source == "not_found"
        assert result.abstract == "The abstract text."

    def test_abstract_fallback_when_no_pdf_available(self):
        """When paper found but no OA PDF and no DOI, fall back to abstract."""
        cache = _make_cache()
        er = _make_existence(doi=None, oa_url=None, abstract="Fallback abstract.")
        mock_client = AsyncMock()

        # Patch the new S2/arXiv fallbacks to no-op so we exercise the
        # abstract_only path. The fallback paths have their own dedicated
        # test coverage in test_fulltext_fallbacks.py.
        with patch(
            "src.verification.api_clients.fulltext._try_s2_fallback_pdf",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.fulltext._try_arxiv_fallback_pdf",
            new=AsyncMock(return_value=None),
        ):
            result = _run(get_full_text(er, mock_client, cache))
        assert result.source == "abstract_only"
        assert result.abstract == "Fallback abstract."


# ---- PDF extraction (GROBID required) ----

class TestPdfExtraction:
    def test_returns_not_found_when_grobid_unavailable(self):
        """Without GROBID running, should return not_found, not crash."""
        from src.verification.api_clients import fulltext as ft_module
        ft_module._reset_grobid_state_for_tests()
        with patch(
            "src.verification.api_clients.fulltext.requests.get",
            side_effect=ConnectionError("GROBID offline for this test"),
        ):
            result = _run(_extract_from_pdf("/nonexistent/path.pdf", source="user_pdf"))
        ft_module._reset_grobid_state_for_tests()
        assert result.source == "not_found"
        assert result.full_text is None


# ---- Async GROBID concurrency guarantees ----
#
# These tests pin the two pieces of concurrency behavior we rely on in the
# agentic pre-retrieve hot path:
#   1. The bounded semaphore actually parallelizes extractions instead of
#      serializing them through a sync `requests.post` on the event loop.
#   2. The availability probe runs exactly once even when many async tasks
#      hit it simultaneously.

class TestGrobidConcurrency:
    def test_extract_runs_in_parallel_up_to_semaphore(self):
        """8 extractions, semaphore=4, each sleeping 0.3s.

        If serialized: ~2.4s. If fully parallel: ~0.3s. With cap=4: ~0.6s.
        We assert the cap-bounded window so a regression to serial work
        (the bug that motivated this change) fails the test loudly.
        """
        from src.verification.api_clients import fulltext as ft_module
        import time

        ft_module._reset_grobid_state_for_tests()

        sleep_s = 0.3
        n_calls = 8
        cap = 4

        # Mark GROBID available so _ensure_grobid_available short-circuits.
        ft_module._grobid_available = True
        # Force a fresh semaphore at the test cap.
        ft_module._grobid_extract_semaphore = asyncio.Semaphore(cap)

        def _fake_extract(_pdf_path):
            time.sleep(sleep_s)
            return ("body text", [{"name": "S", "text": "body text"}])

        with patch(
            "src.verification.api_clients.fulltext._extract_via_grobid",
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

        # Lower bound: even at full concurrency, n_calls/cap batches × sleep_s
        # is the floor. Allow 30% slack for scheduler overhead.
        min_expected = (n_calls / cap) * sleep_s * 0.7
        # Upper bound: must beat fully-serial wall time by a wide margin.
        max_expected = (n_calls / cap) * sleep_s * 2.0
        assert min_expected <= elapsed <= max_expected, (
            f"elapsed={elapsed:.3f}s outside expected range "
            f"[{min_expected:.3f}, {max_expected:.3f}] for cap={cap}"
        )

    def test_availability_probe_runs_once_under_concurrency(self):
        """10 concurrent first-callers must trigger the health probe once."""
        from src.verification.api_clients import fulltext as ft_module

        ft_module._reset_grobid_state_for_tests()

        call_count = 0

        def _fake_health(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.status_code = 200
            return response

        with patch(
            "src.verification.api_clients.fulltext.requests.get",
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
