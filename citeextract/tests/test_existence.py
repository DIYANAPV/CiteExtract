
import asyncio
import os

import httpx
import pytest

from citeextract.models.reference import Reference
from citeextract.verification.api_clients import crossref, semantic_scholar, openalex, pubmed
from citeextract.verification.cache import APICache
from citeextract.verification.existence import check_existence, check_all_references


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.network
class TestCrossRef:
    def test_known_doi(self):
        async def _test():
            async with httpx.AsyncClient() as client:
                return await crossref.lookup_doi("10.1038/nature12373", client)
        result = _run(_test())
        if result is None:
            pytest.skip("CrossRef rate-limited or unreachable")
        assert len(result["title"]) > 5
        assert result["retraction_status"] is False

    def test_fake_doi(self):
        async def _test():
            async with httpx.AsyncClient() as client:
                return await crossref.lookup_doi("10.9999/fake.doi.000", client)
        assert _run(_test()) is None


@pytest.mark.network
class TestSemanticScholar:
    def test_title_search(self):
        async def _test():
            async with httpx.AsyncClient() as client:
                return await semantic_scholar.search_by_title(
                    "Attention Is All You Need", client
                )
        result = _run(_test())
        if result is None:
            pytest.skip("S2 rate-limited (no API key) or unreachable")
        assert "attention" in result["title"].lower()
        assert result.get("abstract") is not None

    def test_nonsense_title(self):
        async def _test():
            async with httpx.AsyncClient() as client:
                return await semantic_scholar.search_by_title(
                    "xyzzy foobar baz totally fake paper 999", client
                )
        result = _run(_test())
        assert result is None

    def test_id_lookup(self):
        async def _test():
            async with httpx.AsyncClient() as client:
                return await semantic_scholar.lookup_by_id("ARXIV:1706.03762", client)
        result = _run(_test())
        if result is None:
            pytest.skip("S2 rate-limited or unreachable")
        assert "attention" in result["title"].lower()


@pytest.mark.network
class TestOpenAlex:
    def test_title_search(self):
        async def _test():
            async with httpx.AsyncClient() as client:
                return await openalex.search_by_title(
                    "Attention Is All You Need", client
                )
        result = _run(_test())
        if result is None:
            pytest.skip("OpenAlex rate-limited or unreachable")
        assert "attention" in result["title"].lower()

    def test_abstract_reconstruction(self):
        inv = {"The": [0], "cat": [1], "sat": [2], "on": [3], "the": [4], "mat": [5]}
        assert openalex._reconstruct_abstract(inv) == "The cat sat on the mat"

    def test_empty_inverted_index(self):
        assert openalex._reconstruct_abstract({}) == ""
        assert openalex._reconstruct_abstract(None) == ""


@pytest.mark.network
class TestPubMed:
    def test_medical_paper(self):
        async def _test():
            async with httpx.AsyncClient() as client:
                return await pubmed.search_by_title(
                    "Effectiveness of mRNA COVID-19 vaccines", client
                )
        result = _run(_test())
        assert result is None or isinstance(result, dict)


class TestCache:
    def test_set_and_get(self):
        async def _test():
            cache = APICache(db_path="/tmp/test_citeextract_cache.db")
            await cache.set("test_key_2", {"foo": "bar"}, ttl_seconds=60)
            result = await cache.get("test_key_2")
            await cache.close()
            return result
        assert _run(_test()) == {"foo": "bar"}

    def test_missing_key(self):
        async def _test():
            cache = APICache(db_path="/tmp/test_citeextract_cache.db")
            result = await cache.get("nonexistent_key_xyz")
            await cache.close()
            return result
        assert _run(_test()) is None

    def test_expired_entry(self):
        async def _test():
            cache = APICache(db_path="/tmp/test_citeextract_cache.db")
            await cache.set("expire_test_2", {"data": 1}, ttl_seconds=-1)
            result = await cache.get("expire_test_2")
            await cache.close()
            return result
        assert _run(_test()) is None


class TestBiomedicalSignal:
    def test_nlp_paper_is_not_biomedical(self):
        from citeextract.verification.existence import _has_biomedical_signal
        assert _has_biomedical_signal(
            "Attention is all you need", "Advances in Neural Information Processing Systems"
        ) is False

    def test_clinical_trial_is_biomedical(self):
        from citeextract.verification.existence import _has_biomedical_signal
        assert _has_biomedical_signal(
            "A randomized clinical trial of drug X for cancer", "JAMA"
        ) is True

    def test_venue_alone_can_signal(self):
        from citeextract.verification.existence import _has_biomedical_signal
        assert _has_biomedical_signal(
            "Network effects on referral patterns", "Journal of Medical Internet Research"
        ) is True

    def test_word_boundary_not_substring(self):
        from citeextract.verification.existence import _has_biomedical_signal
        # "general", "generation" contain "gene" as substring — must NOT match.
        assert _has_biomedical_signal(
            "General-purpose pretraining for natural language generation", "ACL"
        ) is False

    def test_no_inputs(self):
        from citeextract.verification.existence import _has_biomedical_signal
        assert _has_biomedical_signal(None, None) is False
        assert _has_biomedical_signal("", "") is False


class TestCascadeOrder:
    """Cluster-2 reorder: for a no-DOI ref, CrossRef-title runs BEFORE
    OpenAlex, which runs BEFORE S2-title. PubMed is skipped on non-biomedical."""

    def test_no_doi_cascade_order(self, monkeypatch, tmp_path):
        from citeextract.verification import existence as ex
        from citeextract.verification.api_clients import (
            arxiv, crossref, openalex, openreview, pubmed, semantic_scholar,
        )

        call_order: list[str] = []

        async def _cr_lookup_doi(doi, client, *, errors=None):
            call_order.append("crossref_doi")
            return None

        async def _cr_search(title, client, *, ref_authors=None, ref_year=None, errors=None):
            call_order.append("crossref_title")
            return None

        async def _oa_search(title, client, *, ref_authors=None, ref_year=None, errors=None):
            call_order.append("openalex")
            return None

        async def _s2_search(title, client, *, ref_authors=None, ref_year=None, errors=None):
            call_order.append("s2_title")
            return None

        async def _s2_id(s2_id, client, *, errors=None):
            call_order.append("s2_id")
            return None

        async def _pm_search(title, client, *, errors=None):
            call_order.append("pubmed")
            return None

        async def _ax_search(title, client, *, errors=None):
            call_order.append("arxiv_title")
            return None

        async def _ax_authors(authors, year, title, client, *, errors=None):
            call_order.append("arxiv_authors_year")
            return None

        async def _or_search(title, client, *, errors=None):
            call_order.append("openreview")
            return None

        monkeypatch.setattr(crossref, "lookup_doi", _cr_lookup_doi)
        monkeypatch.setattr(crossref, "search_by_title", _cr_search)
        monkeypatch.setattr(openalex, "search_by_title", _oa_search)
        monkeypatch.setattr(semantic_scholar, "search_by_title", _s2_search)
        monkeypatch.setattr(semantic_scholar, "lookup_by_id", _s2_id)
        monkeypatch.setattr(pubmed, "search_by_title", _pm_search)
        monkeypatch.setattr(arxiv, "search_by_title", _ax_search)
        monkeypatch.setattr(arxiv, "search_by_authors_year", _ax_authors)
        monkeypatch.setattr(openreview, "search_by_title", _or_search)

        async def _test():
            ref = Reference(
                ref_id="cascade-test-1",
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer"],
                year=2017,
                source_format="text",
            )
            cache = APICache(db_path=str(tmp_path / "cascade_order.db"))
            async with httpx.AsyncClient() as client:
                result = await check_existence(ref, client, cache)
            await cache.close()
            return result

        result = _run(_test())
        # No DOI / no arxiv_id → S2-by-id should NOT have run
        assert "s2_id" not in call_order
        # crossref_title before openalex before s2_title
        assert "crossref_title" in call_order
        assert "openalex" in call_order
        assert "s2_title" in call_order
        assert call_order.index("crossref_title") < call_order.index("openalex")
        assert call_order.index("openalex") < call_order.index("s2_title")
        # pubmed gated out on non-biomedical title
        assert "pubmed" not in call_order
        assert result.status == "NOT_FOUND"

    def test_pubmed_runs_for_biomedical_title(self, monkeypatch, tmp_path):
        from citeextract.verification import existence as ex
        from citeextract.verification.api_clients import (
            arxiv, crossref, openalex, openreview, pubmed, semantic_scholar,
        )

        called: list[str] = []

        async def _noop(*args, **kw):
            called.append("noop")
            return None

        async def _pm(title, client, *, errors=None):
            called.append("pubmed")
            return None

        monkeypatch.setattr(crossref, "lookup_doi", _noop)
        monkeypatch.setattr(crossref, "search_by_title", _noop)
        monkeypatch.setattr(openalex, "search_by_title", _noop)
        monkeypatch.setattr(semantic_scholar, "search_by_title", _noop)
        monkeypatch.setattr(semantic_scholar, "lookup_by_id", _noop)
        monkeypatch.setattr(pubmed, "search_by_title", _pm)
        monkeypatch.setattr(arxiv, "search_by_title", _noop)
        monkeypatch.setattr(arxiv, "search_by_authors_year", _noop)
        monkeypatch.setattr(openreview, "search_by_title", _noop)

        async def _test():
            ref = Reference(
                ref_id="pubmed-test-1",
                title="A randomized clinical trial of drug X for cancer treatment",
                authors=["Smith"],
                year=2020,
                source_format="text",
            )
            cache = APICache(db_path=str(tmp_path / "cascade_pubmed.db"))
            async with httpx.AsyncClient() as client:
                await check_existence(ref, client, cache)
            await cache.close()

        _run(_test())
        assert "pubmed" in called


@pytest.mark.network
class TestCascade:
    def test_fabricated_paper(self):
        ref = Reference(
            ref_id="fake1",
            title="Quantum Blockchain Neural Networks for Telepathic Communication",
            authors=["Nonexistent Author"],
            year=2099,
            raw_text="Nonexistent (2099). Fake paper.",
            source_format="test",
        )
        result = _run(check_all_references([ref]))[0]
        assert result.status == "NOT_FOUND"
        assert len(result.databases_checked) >= 2
        assert "not_found_in_any_database" in result.flags

    def test_known_paper_by_title(self):
        ref = Reference(
            ref_id="real1",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year=2017,
            raw_text="Vaswani et al. (2017).",
            source_format="test",
        )
        result = _run(check_all_references([ref]))[0]
        if result.status == "NOT_FOUND":
            pytest.skip("All APIs rate-limited; known paper not found (expected in CI)")
        assert result.status == "FOUND"
        assert result.matched_title is not None

    def test_parallel_returns_correct_count(self):
        refs = [
            Reference(
                ref_id=f"p{i}",
                title=f"Fake Paper Number {i} That Does Not Exist",
                authors=["Nobody"],
                year=2099,
                raw_text=f"Nobody (2099). Fake {i}.",
                source_format="test",
            )
            for i in range(3)
        ]
        results = _run(check_all_references(refs))
        assert len(results) == 3
        assert all(r.status == "NOT_FOUND" for r in results)

    def test_known_paper_with_real_doi(self):
        ref = Reference(
            ref_id="doi1",
            title="Nanometre-scale thermometry in a living cell",
            doi="10.1038/nature12373",
            authors=[],
            year=2013,
            raw_text="Kucsko et al. (2013).",
            source_format="test",
        )
        result = _run(check_all_references([ref]))[0]
        if result.status == "NOT_FOUND":
            pytest.skip("CrossRef unreachable")
        assert result.status == "FOUND"
        assert result.source == "crossref"
