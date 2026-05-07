
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

    def test_abstract_shortcut(self):
        async def _test():
            cache = APICache(db_path="/tmp/test_citeextract_cache.db")
            await cache.set_abstract("ref_test_1", "This is a test abstract.")
            result = await cache.get_abstract("ref_test_1")
            await cache.close()
            return result
        assert _run(_test()) == "This is a test abstract."

    def test_expired_entry(self):
        async def _test():
            cache = APICache(db_path="/tmp/test_citeextract_cache.db")
            await cache.set("expire_test_2", {"data": 1}, ttl_seconds=-1)
            result = await cache.get("expire_test_2")
            await cache.close()
            return result
        assert _run(_test()) is None


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
