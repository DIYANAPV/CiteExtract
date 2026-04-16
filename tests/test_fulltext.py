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

        result = _run(get_full_text(er, mock_client, cache))
        assert result.source == "abstract_only"
        assert result.abstract == "Fallback abstract."


# ---- PDF extraction (GROBID required) ----

class TestPdfExtraction:
    def test_returns_not_found_when_grobid_unavailable(self):
        """Without GROBID running, should return not_found, not crash."""
        result = _extract_from_pdf("/nonexistent/path.pdf", source="user_pdf")
        assert result.source == "not_found"
        assert result.full_text is None
