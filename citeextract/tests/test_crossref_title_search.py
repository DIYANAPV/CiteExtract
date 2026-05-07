
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from citeextract.verification.api_clients import crossref


def _run(coro):
    return asyncio.run(coro)


def _crossref_response(items: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"message": {"items": items}}
    return resp


def _item(
    *, title: str, doi: str, type: str = "journal-article",
    authors: list[dict] = None, year: int = 2023,
    container: str = "Some Journal",
) -> dict:
    return {
        "title": [title],
        "DOI": doi,
        "type": type,
        "author": authors or [{"given": "Jane", "family": "Doe"}],
        "published-print": {"date-parts": [[year]]},
        "container-title": [container],
    }


class TestSearchByTitle:
    def test_returns_high_quality_match(self):
        client = AsyncMock()
        items = [_item(
            title="Attention Is All You Need",
            doi="10.5555/aiayn",
            type="journal-article",
        )]
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(return_value=_crossref_response(items)),
        ):
            result = _run(crossref.search_by_title("Attention Is All You Need", client))
        assert result is not None
        assert result["doi"] == "10.5555/aiayn"
        assert result["title"] == "Attention Is All You Need"
        assert result["type"] == "journal-article"

    def test_prefers_journal_article_over_reference_entry(self):
        client = AsyncMock()
        items = [
            _item(title="Perceptron", doi="10.1/encyclo", type="reference-entry"),
            _item(title="Perceptron", doi="10.1/journal", type="journal-article"),
        ]
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(return_value=_crossref_response(items)),
        ):
            result = _run(crossref.search_by_title("Perceptron", client))
        assert result is not None
        assert result["doi"] == "10.1/journal"

    def test_falls_back_to_reference_entry_when_only_option(self):
        client = AsyncMock()
        items = [_item(
            title="Perceptron",
            doi="10.1/encyclo",
            type="reference-entry",
        )]
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(return_value=_crossref_response(items)),
        ):
            result = _run(crossref.search_by_title("Perceptron", client))
        assert result is not None
        assert result["doi"] == "10.1/encyclo"

    def test_returns_none_when_no_item_clears_threshold(self):
        client = AsyncMock()
        items = [_item(
            title="Completely Unrelated Topic",
            doi="10.1/wrong",
        )]
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(return_value=_crossref_response(items)),
        ):
            result = _run(crossref.search_by_title("Attention Is All You Need", client))
        assert result is None

    def test_short_title_skips_search(self):
        client = AsyncMock()
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(return_value=_crossref_response([])),
        ) as fetcher:
            result = _run(crossref.search_by_title("ab", client))
        assert result is None
        fetcher.assert_not_awaited()

    def test_handles_empty_response(self):
        client = AsyncMock()
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(return_value=_crossref_response([])),
        ):
            result = _run(crossref.search_by_title("Any Title", client))
        assert result is None

    def test_handles_http_error(self):
        client = AsyncMock()
        resp = MagicMock(); resp.status_code = 500
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(return_value=resp),
        ):
            result = _run(crossref.search_by_title("Some Title", client))
        assert result is None

    def test_handles_request_exception(self):
        client = AsyncMock()
        with patch(
            "citeextract.verification.api_clients.crossref.fetch_with_retry",
            new=AsyncMock(side_effect=httpx.RequestError("boom")),
        ):
            result = _run(crossref.search_by_title("Some Title", client))
        assert result is None
