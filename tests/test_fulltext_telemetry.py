"""Tests for full-text retrieval telemetry — every step of the waterfall is
visible on ``FullTextResult.attempts`` so silent fallthroughs to abstract_only
are diagnosable from the report alone.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.comprehension import FetchAttempt, FullTextResult
from src.models.verdict import ExistenceResult
from src.verification.api_clients.fulltext import (
    _download_and_extract,
    _cache_fulltext,
    _extract_citation_pdf_url,
    _FULLTEXT_CACHE_MAX_CHARS,
    get_full_text,
)


def _run(coro):
    return asyncio.run(coro)


def _make_existence(
    ref_id: str = "1",
    status: str = "FOUND",
    doi: str = "10.1234/test",
    oa_url: str = None,
    arxiv_id: str = None,
    title: str = "Test Paper Title",
    abstract: str = "Abstract.",
) -> ExistenceResult:
    return ExistenceResult(
        ref_id=ref_id,
        status=status,
        source="semantic_scholar",
        matched_title=title,
        matched_authors=["Smith"],
        matched_year=2023,
        matched_doi=doi,
        matched_arxiv_id=arxiv_id,
        abstract=abstract,
        oa_url=oa_url,
        databases_checked=["semantic_scholar"],
    )


def _make_cache():
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


def _mock_response(
    *, status_code: int, content_type: str = "application/pdf",
    content: bytes = b"%PDF-1.4 fake", url: str = "https://example.com/x.pdf",
    retry_after: str = None,
):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    if retry_after is not None:
        resp.headers["Retry-After"] = retry_after
    resp.content = content
    resp.url = url
    return resp


class TestDownloadAndExtractAttempt:
    """``_download_and_extract`` returns ``(result, list[FetchAttempt])``.

    The list is normally one element; the HTML rescue path adds a second.
    """

    def test_records_ok_on_success(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(status_code=200))
        with patch(
            "src.verification.api_clients.fulltext._extract_from_pdf",
            new=AsyncMock(return_value=FullTextResult(
                source="oa_url", full_text="body", sections=[],
            )),
        ):
            result, atts = _run(_download_and_extract(
                "https://example.com/p.pdf", client, source="oa_url",
            ))
        assert result is not None and result.full_text == "body"
        assert len(atts) == 1
        assert atts[0].source == "oa_url"
        assert atts[0].status == "ok"
        assert atts[0].ms >= 0

    def test_records_http_error_on_404(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(status_code=404))
        result, atts = _run(_download_and_extract(
            "https://example.com/p.pdf", client, source="oa_url",
        ))
        assert result is None
        assert len(atts) == 1
        assert atts[0].status == "http_error"
        assert atts[0].detail == "http_404"

    def test_records_not_pdf_on_html_response(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(
            status_code=200, content_type="text/html",
        ))
        result, atts = _run(_download_and_extract(
            "https://example.com/p.pdf", client, source="oa_url",
        ))
        assert result is None
        assert len(atts) == 1
        assert atts[0].status == "not_pdf"
        assert atts[0].detail.startswith("content-type:text/html")

    def test_records_grobid_failed_when_extract_returns_no_text(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(status_code=200))
        with patch(
            "src.verification.api_clients.fulltext._extract_from_pdf",
            new=AsyncMock(return_value=FullTextResult(
                source="not_found", full_text=None,
            )),
        ):
            result, atts = _run(_download_and_extract(
                "https://example.com/p.pdf", client, source="oa_url",
            ))
        assert result is not None and result.full_text is None
        assert len(atts) == 1
        assert atts[0].status == "grobid_failed"

    def test_honors_retry_after_then_succeeds(self):
        """429 with Retry-After: 0 should retry quickly and surface ok."""
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[
            _mock_response(status_code=429, retry_after="0"),
            _mock_response(status_code=200),
        ])
        with patch(
            "src.verification.api_clients.fulltext._extract_from_pdf",
            new=AsyncMock(return_value=FullTextResult(
                source="arxiv", full_text="body",
            )),
        ):
            result, atts = _run(_download_and_extract(
                "https://arxiv.org/p.pdf", client, source="arxiv",
            ))
        assert result is not None and result.full_text == "body"
        assert len(atts) == 1
        assert atts[0].status == "ok"
        assert client.get.await_count == 2


class TestAttemptsOnFullTextResult:
    """End-to-end: ``get_full_text`` populates ``attempts`` on its return."""

    def test_attempts_empty_when_cache_hit(self):
        """Cache-hit path doesn't replay attempts (the cache stores them
        on the entry it returns — the test asserts that cached deserialization
        still works regardless of attempts)."""
        cache = _make_cache()
        cache.get = AsyncMock(return_value={
            "source": "s2_api", "full_text": "cached", "sections": [],
            "abstract": None, "truncated": False, "attempts": [],
        })
        result = _run(get_full_text(_make_existence(), AsyncMock(), cache))
        assert result.full_text == "cached"
        assert result.attempts == []

    def test_attempts_recorded_for_oa_url_success(self):
        cache = _make_cache()
        client = AsyncMock()
        er = _make_existence(oa_url="https://example.com/p.pdf")
        with patch(
            "src.verification.api_clients.fulltext._download_and_extract",
            new=AsyncMock(return_value=(
                FullTextResult(source="oa_url", full_text="body"),
                [FetchAttempt(source="oa_url", status="ok", ms=12)],
            )),
        ):
            result = _run(get_full_text(er, client, cache))
        assert result.full_text == "body"
        assert len(result.attempts) == 1
        assert result.attempts[0].status == "ok"
        # The public-facing source field reflects the actual path that
        # succeeded (``oa_url`` here, ``html_meta_pdf`` when we rescued
        # via a landing page) — no relabel that hides the recovery.
        assert result.source == "oa_url"

    def test_attempts_recorded_for_abstract_only_fallthrough(self):
        """When every fetch path fails or is skipped, the attempts list
        carries the full trace and the result is abstract_only."""
        cache = _make_cache()
        client = AsyncMock()
        er = _make_existence(
            doi="10.1234/journal", oa_url="https://example.com/p.pdf",
            abstract="The abstract.",
        )
        # Make oa_url and unpaywall fetches fail; force fallbacks to no-op.
        with patch(
            "src.verification.api_clients.fulltext._download_and_extract",
            new=AsyncMock(return_value=(
                None,
                [FetchAttempt(source="oa_url", status="not_pdf", detail="content-type:text/html")],
            )),
        ), patch(
            "src.verification.api_clients.fulltext._unpaywall_pdf_url",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.fulltext._try_s2_fallback_pdf",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.verification.api_clients.fulltext._try_arxiv_fallback_pdf",
            new=AsyncMock(return_value=None),
        ):
            result = _run(get_full_text(er, client, cache))
        assert result.source == "abstract_only"
        sources = [a.source for a in result.attempts]
        assert "oa_url" in sources
        assert "unpaywall" in sources
        assert "abstract_only" in sources

    def test_attempts_recorded_for_arxiv_path(self):
        cache = _make_cache()
        client = AsyncMock()
        er = _make_existence(
            doi=None, oa_url=None, arxiv_id="2204.02311",
        )
        with patch(
            "src.verification.api_clients.fulltext._download_and_extract_arxiv",
            new=AsyncMock(return_value=(
                FullTextResult(source="arxiv", full_text="body"),
                [FetchAttempt(source="arxiv", status="ok", ms=42)],
            )),
        ):
            result = _run(get_full_text(er, client, cache))
        assert result.full_text == "body"
        sources = [a.source for a in result.attempts]
        assert sources == ["arxiv"]


# ---- Per-host concurrency cap ----
#
# These tests pin the behavior that prevents the arXiv 429-storm: 13+
# concurrent downloads must not all hit arxiv.org at once. Without the
# semaphore, the regression is a silent fall-through to abstract_only —
# very hard to catch via unit tests on individual call sites alone.

class TestPdfHostConcurrency:
    def test_arxiv_downloads_capped_at_host_limit(self):
        """8 simultaneous arxiv.org downloads must not exceed the host cap."""
        from src.verification.api_clients import fulltext as ft_module

        ft_module._reset_pdf_host_semaphores_for_tests()
        ft_module._reset_grobid_state_for_tests()
        ft_module._grobid_available = True

        in_flight = 0
        max_in_flight = 0

        async def fake_get(*_args, **_kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return _mock_response(status_code=200, url="https://export.arxiv.org/pdf/x")

        client = AsyncMock()
        client.get = fake_get

        with patch(
            "src.verification.api_clients.fulltext._extract_from_pdf",
            new=AsyncMock(return_value=FullTextResult(
                source="arxiv", full_text="body",
            )),
        ):
            async def _all():
                return await asyncio.gather(*[
                    _download_and_extract(
                        f"https://export.arxiv.org/pdf/250{i}.0001",
                        client, source="arxiv",
                    )
                    for i in range(8)
                ])

            results = _run(_all())

        ft_module._reset_pdf_host_semaphores_for_tests()
        ft_module._reset_grobid_state_for_tests()

        cap = ft_module._PDF_HOST_CONCURRENCY["export.arxiv.org"]
        assert all(r is not None and r[0].full_text for r in results)
        assert max_in_flight <= cap, (
            f"observed {max_in_flight} concurrent downloads, expected <= {cap}"
        )

    def test_unknown_host_uses_default_cap(self):
        """A host without a configured cap gets the default."""
        from src.verification.api_clients import fulltext as ft_module

        ft_module._reset_pdf_host_semaphores_for_tests()
        sem = ft_module._get_pdf_host_semaphore("https://random.example.com/p.pdf")
        assert sem._value == ft_module._PDF_DEFAULT_CONCURRENCY
        ft_module._reset_pdf_host_semaphores_for_tests()

    def test_arxiv_alias_hosts_share_semaphore(self):
        """arxiv.org and export.arxiv.org both have the same per-host budget,
        but they are different hostnames — assert each gets its own semaphore
        with the configured cap (no cross-host leakage either direction)."""
        from src.verification.api_clients import fulltext as ft_module

        ft_module._reset_pdf_host_semaphores_for_tests()
        s1 = ft_module._get_pdf_host_semaphore("https://arxiv.org/pdf/x")
        s2 = ft_module._get_pdf_host_semaphore("https://export.arxiv.org/pdf/x")
        assert s1._value == ft_module._PDF_HOST_CONCURRENCY["arxiv.org"]
        assert s2._value == ft_module._PDF_HOST_CONCURRENCY["export.arxiv.org"]
        ft_module._reset_pdf_host_semaphores_for_tests()


class TestLongPaperCacheTruncation:
    """Long papers (PaLM, Llama 3, OLMo) used to be poisoned by the 200K cap.
    The new soft-truncation path keeps them as full_text with a flag."""

    def test_under_cap_paper_cached_intact(self):
        """A 100K-char paper (well under the cap) goes in unchanged."""
        cache = _make_cache()
        body = "x" * 100_000
        result = FullTextResult(
            source="arxiv", full_text=body, sections=[{"name": "S", "text": body}],
        )
        _run(_cache_fulltext(cache, "fulltext_v2:arxiv:test", "1", result))
        cache.set.assert_awaited_once()
        _, payload, _ttl = cache.set.await_args[0]
        assert payload["full_text"] == body
        assert payload["truncated"] is False
        assert payload["source"] == "arxiv"

    def test_over_cap_paper_truncated_not_dropped(self):
        """A paper longer than the cap gets sliced and flagged, not nulled."""
        cache = _make_cache()
        body = "y" * (_FULLTEXT_CACHE_MAX_CHARS + 50_000)
        result = FullTextResult(source="arxiv", full_text=body, sections=[])
        _run(_cache_fulltext(cache, "fulltext_v2:arxiv:big", "1", result))
        _, payload, _ttl = cache.set.await_args[0]
        assert payload["full_text"] is not None, (
            "regression: long papers got cached as not_found again"
        )
        assert len(payload["full_text"]) == _FULLTEXT_CACHE_MAX_CHARS
        assert payload["truncated"] is True
        assert payload["source"] == "arxiv"

    def test_over_cap_paper_sections_pruned(self):
        """Sections past the truncation point are dropped (no duplicate body)."""
        cache = _make_cache()
        big_section = {"name": "Body", "text": "z" * (_FULLTEXT_CACHE_MAX_CHARS + 1)}
        small_section = {"name": "Conclusion", "text": "thanks"}
        result = FullTextResult(
            source="arxiv",
            full_text="z" * (_FULLTEXT_CACHE_MAX_CHARS + 50_000),
            sections=[big_section, small_section],
        )
        _run(_cache_fulltext(cache, "fulltext_v2:arxiv:big", "1", result))
        _, payload, _ttl = cache.set.await_args[0]
        # First section already exceeded the cap on its own, so the kept
        # list stops before adding it (running_len + len > cap).
        assert payload["sections"] == []

    def test_no_op_when_full_text_missing(self):
        """An abstract-only result skips the truncation branch entirely."""
        cache = _make_cache()
        result = FullTextResult(source="abstract_only", abstract="abc")
        _run(_cache_fulltext(cache, "fulltext_v2:title:abc", "1", result))
        _, payload, _ttl = cache.set.await_args[0]
        assert payload["full_text"] is None
        assert payload["truncated"] is False


class TestHtmlMetaPdfRescue:
    """Institutional landing pages (HAL, ResearchGate) serve HTML; the rescue
    path follows the ``citation_pdf_url`` meta tag to the actual PDF."""

    def test_extract_handles_standard_attribute_order(self):
        html = b'<html><head><meta name="citation_pdf_url" content="https://x.org/p.pdf"></head></html>'
        assert _extract_citation_pdf_url(
            html, "https://hal.science/foo",
        ) == "https://x.org/p.pdf"

    def test_extract_handles_reversed_attribute_order(self):
        html = b'<meta content="https://x.org/p.pdf" name="citation_pdf_url">'
        assert _extract_citation_pdf_url(
            html, "https://hal.science/foo",
        ) == "https://x.org/p.pdf"

    def test_extract_resolves_protocol_relative_url(self):
        html = b'<meta name="citation_pdf_url" content="//cdn.repo.edu/p.pdf">'
        assert _extract_citation_pdf_url(
            html, "https://repo.edu/landing",
        ) == "https://cdn.repo.edu/p.pdf"

    def test_extract_resolves_host_relative_url(self):
        html = b'<meta name="citation_pdf_url" content="/files/paper.pdf">'
        assert _extract_citation_pdf_url(
            html, "https://hal.science/landing/foo",
        ) == "https://hal.science/files/paper.pdf"

    def test_extract_returns_none_when_meta_absent(self):
        html = b'<html><head><title>Some unrelated page</title></head></html>'
        assert _extract_citation_pdf_url(html, "https://x.org") is None

    def test_html_landing_page_is_recursed_into(self):
        """End-to-end: HTML response with citation_pdf_url leads to a
        successful PDF fetch via the html_meta_pdf path."""
        from src.verification.api_clients import fulltext as ft_module

        ft_module._reset_pdf_host_semaphores_for_tests()
        ft_module._reset_grobid_state_for_tests()
        ft_module._grobid_available = True

        landing = _mock_response(
            status_code=200,
            content_type="text/html; charset=UTF-8",
            content=(
                b'<html><head><meta name="citation_pdf_url" '
                b'content="https://export.arxiv.org/pdf/2407.21783"></head></html>'
            ),
            url="https://hal.science/hal-05414211",
        )
        pdf = _mock_response(
            status_code=200, content_type="application/pdf",
            url="https://export.arxiv.org/pdf/2407.21783",
        )
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[landing, pdf])

        with patch(
            "src.verification.api_clients.fulltext._extract_from_pdf",
            new=AsyncMock(return_value=FullTextResult(
                source="html_meta_pdf", full_text="body",
            )),
        ):
            result, atts = _run(_download_and_extract(
                "https://hal.science/hal-05414211", client, source="oa_url",
            ))

        ft_module._reset_pdf_host_semaphores_for_tests()
        ft_module._reset_grobid_state_for_tests()

        assert result is not None and result.full_text == "body"
        # Both hops are visible in the trace: the original oa_url GET that
        # returned HTML (``not_pdf``) AND the rescued html_meta_pdf GET
        # that succeeded. Without the chain, debugging "where did the PDF
        # come from?" requires log spelunking.
        assert len(atts) == 2
        assert atts[0].source == "oa_url"
        assert atts[0].status == "not_pdf"
        assert atts[1].source == "html_meta_pdf"
        assert atts[1].status == "ok"
        assert client.get.await_count == 2

    def test_recursion_does_not_loop_when_html_lacks_meta(self):
        """HTML page without citation_pdf_url falls through to not_pdf without
        recursing or hammering the server."""
        from src.verification.api_clients import fulltext as ft_module

        ft_module._reset_pdf_host_semaphores_for_tests()

        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(
            status_code=200,
            content_type="text/html",
            content=b'<html><body>nothing useful</body></html>',
            url="https://hal.science/foo",
        ))
        result, atts = _run(_download_and_extract(
            "https://hal.science/foo", client, source="oa_url",
        ))

        ft_module._reset_pdf_host_semaphores_for_tests()

        assert result is None
        assert len(atts) == 1
        assert atts[0].status == "not_pdf"
        assert client.get.await_count == 1, "should not recurse without a meta tag"


class TestRetryAfterHandling:
    def test_429_with_retry_after_header_waits_then_succeeds(self):
        """``Retry-After: 1`` should be honored; result.attempts records ok."""
        from src.verification.api_clients import fulltext as ft_module
        import time as time_module

        ft_module._reset_pdf_host_semaphores_for_tests()
        ft_module._reset_grobid_state_for_tests()
        ft_module._grobid_available = True

        client = AsyncMock()
        client.get = AsyncMock(side_effect=[
            _mock_response(status_code=429, retry_after="1"),
            _mock_response(status_code=200),
        ])
        with patch(
            "src.verification.api_clients.fulltext._extract_from_pdf",
            new=AsyncMock(return_value=FullTextResult(
                source="arxiv", full_text="body",
            )),
        ):
            t0 = time_module.perf_counter()
            result, atts = _run(_download_and_extract(
                "https://export.arxiv.org/pdf/x", client, source="arxiv",
            ))
            elapsed = time_module.perf_counter() - t0

        ft_module._reset_pdf_host_semaphores_for_tests()
        ft_module._reset_grobid_state_for_tests()

        assert result is not None and result.full_text == "body"
        assert len(atts) == 1
        assert atts[0].status == "ok"
        # Retry-After: 1 → at least ~1s wait. _PDF_BASE_DELAY is 2s on first
        # retry so the actual floor is 2s. Allow some slack for scheduler.
        assert elapsed >= 1.0, (
            f"Retry-After not honored — elapsed={elapsed:.2f}s, expected >= 1.0s"
        )
