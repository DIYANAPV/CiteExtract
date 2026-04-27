"""Unit tests for the OpenReview client and its insertion into the L2 cascade.

The OpenReview fallback closes the gap for OpenReview-only tech reports
like LeCun's "A Path Towards Autonomous Machine Intelligence" (forum id
``BZ5a1r-kVsf``), which has no DOI in CrossRef and no arXiv preprint.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.verification.api_clients import openreview


def _run(coro):
    return asyncio.run(coro)


def _openreview_response(notes: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"notes": notes, "count": len(notes)}
    return resp


def _note(
    *, forum_id: str, title: str | None,
    authors=None, abstract: str | None = None,
    cdate: int | None = None,
) -> dict:
    content: dict = {}
    if title is not None:
        content["title"] = title
    if authors is not None:
        content["authors"] = authors
    if abstract is not None:
        content["abstract"] = abstract
    return {"id": forum_id, "forum": forum_id, "content": content, "cdate": cdate}


class TestSearchByTitle:
    def test_returns_high_quality_match(self):
        client = AsyncMock()
        notes = [_note(
            forum_id="BZ5a1r-kVsf",
            title="A Path Towards Autonomous Machine Intelligence",
            authors=["Yann LeCun"],
        )]
        with patch(
            "src.verification.api_clients.openreview.fetch_with_retry",
            new=AsyncMock(return_value=_openreview_response(notes)),
        ):
            result = _run(openreview.search_by_title(
                "A Path Towards Autonomous Machine Intelligence", client,
            ))
        assert result is not None
        assert result["title"] == "A Path Towards Autonomous Machine Intelligence"
        assert result["authors"] == ["Yann LeCun"]
        assert result["oa_url"] == "https://openreview.net/pdf?id=BZ5a1r-kVsf"

    def test_skips_notes_without_title(self):
        """OpenReview indexes reviews and comments without titles — the
        client must skip them, not crash."""
        client = AsyncMock()
        notes = [
            _note(forum_id="abc", title=None),  # comment / review
            _note(
                forum_id="xyz",
                title="Some Random Paper About Things",
            ),
        ]
        with patch(
            "src.verification.api_clients.openreview.fetch_with_retry",
            new=AsyncMock(return_value=_openreview_response(notes)),
        ):
            result = _run(openreview.search_by_title(
                "Some Random Paper About Things", client,
            ))
        assert result is not None
        assert result["title"] == "Some Random Paper About Things"

    def test_returns_none_when_nothing_clears_threshold(self):
        client = AsyncMock()
        notes = [_note(
            forum_id="abc", title="Completely Different Subject",
        )]
        with patch(
            "src.verification.api_clients.openreview.fetch_with_retry",
            new=AsyncMock(return_value=_openreview_response(notes)),
        ):
            result = _run(openreview.search_by_title(
                "A Path Towards Autonomous Machine Intelligence", client,
            ))
        assert result is None

    def test_handles_v2_value_envelope(self):
        """v2 entries wrap content in ``{"value": "..."}`` envelopes; the
        client handles either shape."""
        client = AsyncMock()
        notes = [{
            "id": "v2",
            "forum": "v2",
            "content": {
                "title": {"value": "A Path Towards Autonomous Machine Intelligence"},
                "authors": {"value": ["Yann LeCun"]},
            },
        }]
        with patch(
            "src.verification.api_clients.openreview.fetch_with_retry",
            new=AsyncMock(return_value=_openreview_response(notes)),
        ):
            result = _run(openreview.search_by_title(
                "A Path Towards Autonomous Machine Intelligence", client,
            ))
        assert result is not None
        assert result["title"] == "A Path Towards Autonomous Machine Intelligence"
        assert result["authors"] == ["Yann LeCun"]

    def test_short_title_skips_search(self):
        client = AsyncMock()
        with patch(
            "src.verification.api_clients.openreview.fetch_with_retry",
            new=AsyncMock(return_value=_openreview_response([])),
        ) as fetcher:
            result = _run(openreview.search_by_title("ab", client))
        assert result is None
        fetcher.assert_not_awaited()

    def test_handles_http_error(self):
        client = AsyncMock()
        resp = MagicMock(); resp.status_code = 500
        with patch(
            "src.verification.api_clients.openreview.fetch_with_retry",
            new=AsyncMock(return_value=resp),
        ):
            result = _run(openreview.search_by_title("Some Title", client))
        assert result is None


class TestCascadeIntegration:
    """When the experimental flag is off, OpenReview must NOT be consulted —
    the L2 cascade behaviour stays stable for users who don't opt in."""

    def test_flag_off_skips_openreview(self):
        from src.models.reference import Reference
        from src.verification.existence import check_existence
        from src.verification.cache import APICache

        ref = Reference(
            ref_id="t1",
            title="A Path Towards Autonomous Machine Intelligence",
            source_format="text",
        )
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.get_title_ref_id = AsyncMock(return_value=None)
        cache.get_all_title_keys = AsyncMock(return_value=[])
        cache.set_abstract = AsyncMock()
        cache.set_title_index = AsyncMock()

        client = AsyncMock()

        with patch(
            "src.verification.api_clients.semantic_scholar.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.openalex.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.crossref.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.pubmed.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.arxiv.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.config.experimental_fallbacks",
            return_value={"openreview": False},
        ), patch(
            "src.verification.api_clients.openreview.search_by_title",
            new=AsyncMock(side_effect=AssertionError("openreview must not be called")),
        ):
            result = _run(check_existence(ref, client, cache))

        assert result.status == "NOT_FOUND"
        assert "openreview" not in result.databases_checked

    def test_flag_on_consults_openreview(self):
        from src.models.reference import Reference
        from src.verification.existence import check_existence
        from src.verification.cache import APICache

        ref = Reference(
            ref_id="t2",
            title="A Path Towards Autonomous Machine Intelligence",
            source_format="text",
        )
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.get_title_ref_id = AsyncMock(return_value=None)
        cache.get_all_title_keys = AsyncMock(return_value=[])
        cache.set_abstract = AsyncMock()
        cache.set_title_index = AsyncMock()

        client = AsyncMock()

        or_match = {
            "title": "A Path Towards Autonomous Machine Intelligence",
            "authors": ["Yann LeCun"],
            "year": 2022,
            "venue": "OpenReview",
            "abstract": "...",
            "doi": None,
            "arxiv_id": None,
            "oa_url": "https://openreview.net/pdf?id=BZ5a1r-kVsf",
            "title_similarity": 1.0,
        }

        with patch(
            "src.verification.api_clients.semantic_scholar.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.openalex.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.crossref.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.pubmed.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.arxiv.search_by_title",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.config.experimental_fallbacks",
            return_value={"openreview": True},
        ), patch(
            "src.verification.api_clients.openreview.search_by_title",
            new=AsyncMock(return_value=or_match),
        ):
            result = _run(check_existence(ref, client, cache))

        assert result.status == "FOUND"
        assert result.source == "openreview"
        assert "openreview" in result.databases_checked
