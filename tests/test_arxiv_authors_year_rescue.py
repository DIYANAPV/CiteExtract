"""Tests for the arXiv author + year rescue path.

Closes a real-world gap where a paper's arXiv title was renamed across
versions (e.g. arXiv 2505.14376 was uploaded as "AutoRev: Automatic Peer
Review System for Academic Research Papers" in v1, and in v3 became
"Graph-Guided Passage Retrieval for Author-Centric Structured Feedback"
— the API only returns the latest title). The strict title-similarity
check fails, but author surnames + year still uniquely identify the
paper. This rescue path verifies authors and year before accepting.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.verification.api_clients import arxiv
from src.verification.matching import author_containment


def _run(coro):
    return asyncio.run(coro)


def _atom_response(entries: list[dict]) -> MagicMock:
    """Build a mock httpx Response containing an arXiv Atom feed.

    Each entry dict: {title, authors: list[str], year, arxiv_id}.
    """
    entry_xml = []
    for e in entries:
        authors_xml = "".join(f"<author><name>{a}</name></author>" for a in e["authors"])
        entry_xml.append(f"""
        <entry>
            <id>http://arxiv.org/abs/{e['arxiv_id']}v3</id>
            <title>{e['title']}</title>
            <published>{e['year']}-05-20T13:59:58Z</published>
            <summary>Some abstract text.</summary>
            {authors_xml}
        </entry>
        """)
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        + "".join(entry_xml) +
        '</feed>'
    )
    resp = MagicMock()
    resp.status_code = 200
    resp.text = feed
    resp.raise_for_status = MagicMock()
    return resp


class TestSearchByAuthorsYear:
    def test_finds_renamed_paper(self):
        """The AutoRev case: arXiv title is the v3 name, ref title is the
        v1 name, but authors + year line up."""
        client = AsyncMock()

        # arXiv returns the paper under its current (v3) title.
        resp = _atom_response([{
            "title": "Graph-Guided Passage Retrieval for Author-Centric Structured Feedback",
            "authors": [
                "Maitreya Prafulla Chitale",
                "Ketaki Mangesh Shetye",
                "Harshit Gupta",
                "Manav Chaudhary",
                "Manish Shrivastava",
                "Vasudeva Varma",
            ],
            "year": 2025,
            "arxiv_id": "2505.14376",
        }])

        with patch(
            "src.verification.api_clients.arxiv.fetch_with_retry",
            new=AsyncMock(return_value=resp),
        ):
            result = _run(arxiv.search_by_authors_year(
                authors=[
                    "M. P. Chitale", "K. M. Shetye", "H. Gupta",
                    "M. Chaudhary", "V. Varma",
                ],
                year=2025,
                title_hint="AutoRev: Automatic peer review system for academic research papers",
                client=client,
            ))

        assert result is not None, "expected the rescue path to find the renamed paper"
        assert result["arxiv_id"] == "2505.14376"
        assert result.get("title_evolved") is True

    def test_rejects_single_author_refs(self):
        """A reference with only one author is too risky to match — skip
        the rescue rather than risk a wrong paper on a common surname."""
        client = AsyncMock()

        with patch(
            "src.verification.api_clients.arxiv.fetch_with_retry",
            new=AsyncMock(side_effect=AssertionError(
                "fetch_with_retry must not be called when only 1 author"
            )),
        ):
            result = _run(arxiv.search_by_authors_year(
                authors=["Yann LeCun"],
                year=2022,
                title_hint="A Path Towards Autonomous Machine Intelligence",
                client=client,
            ))
        assert result is None

    def test_rejects_when_year_off_by_more_than_one(self):
        """Year drift of >1 is suspicious — likely a wrong paper."""
        client = AsyncMock()
        resp = _atom_response([{
            "title": "Graph-Guided Passage Retrieval for Author-Centric Structured Feedback",
            "authors": ["Maitreya Chitale", "Ketaki Shetye", "Vasudeva Varma"],
            "year": 2025,
            "arxiv_id": "2505.14376",
        }])
        with patch(
            "src.verification.api_clients.arxiv.fetch_with_retry",
            new=AsyncMock(return_value=resp),
        ):
            result = _run(arxiv.search_by_authors_year(
                authors=["M. P. Chitale", "K. M. Shetye", "V. Varma"],
                year=2020,  # 5 years off
                title_hint="AutoRev: Automatic peer review system",
                client=client,
            ))
        assert result is None, "year drift of 5 years should reject the candidate"

    def test_rejects_when_author_overlap_too_low(self):
        """A 3-author ref where only 1 surname matches the candidate
        shouldn't qualify even with strong title hint."""
        client = AsyncMock()
        resp = _atom_response([{
            "title": "Some Other Paper",
            # Only "Smith" overlaps with the ref
            "authors": ["John Smith", "Alice Brown", "Bob Johnson"],
            "year": 2025,
            "arxiv_id": "2505.99999",
        }])
        with patch(
            "src.verification.api_clients.arxiv.fetch_with_retry",
            new=AsyncMock(return_value=resp),
        ):
            result = _run(arxiv.search_by_authors_year(
                authors=["Smith J.", "Williams A.", "Davis B."],
                year=2025,
                title_hint="Some Other Paper",
                client=client,
            ))
        # 1/3 = 0.33 containment, below 0.6 threshold
        assert result is None

    def test_rejects_when_title_completely_unrelated(self):
        """If author + year match but the title hint shares zero overlap
        with the candidate's current title, the floor of 0.30 rejects it."""
        client = AsyncMock()
        resp = _atom_response([{
            "title": "Quantum Computing in Particle Physics",
            "authors": ["Jane Smith", "Bob Lee"],
            "year": 2025,
            "arxiv_id": "2505.11111",
        }])
        with patch(
            "src.verification.api_clients.arxiv.fetch_with_retry",
            new=AsyncMock(return_value=resp),
        ):
            result = _run(arxiv.search_by_authors_year(
                authors=["Smith J.", "Lee B."],
                year=2025,
                title_hint="Bayesian inference for cosmological surveys",
                client=client,
            ))
        # Title sim < 0.30; should reject even with strong author+year
        assert result is None

    def test_handles_http_failure(self):
        client = AsyncMock()
        with patch(
            "src.verification.api_clients.arxiv.fetch_with_retry",
            new=AsyncMock(side_effect=httpx.RequestError("boom")),
        ):
            result = _run(arxiv.search_by_authors_year(
                authors=["Smith J.", "Lee B."],
                year=2025,
                title_hint="Some title",
                client=client,
            ))
        assert result is None


class TestAuthorContainment:
    def test_full_overlap_is_one(self):
        assert author_containment(
            ["Smith", "Jones", "Lee"],
            ["John Smith", "Alice Jones", "Bob Lee"],
        ) == 1.0

    def test_partial_overlap(self):
        assert author_containment(
            ["Smith", "Jones", "Lee"],
            ["John Smith", "Alice Jones"],
        ) == pytest_approx(2/3)

    def test_extra_db_authors_dont_lower_score(self):
        """Author list on candidate is larger — score still 1.0 if all ref
        authors are present (asymmetric, unlike Jaccard)."""
        assert author_containment(
            ["Smith"],
            ["John Smith", "Alice Brown", "Bob Lee", "Carol White"],
        ) == 1.0

    def test_empty_inputs(self):
        assert author_containment([], ["X", "Y"]) == 0.0
        assert author_containment(["X"], []) == 0.0


def pytest_approx(value, tol=1e-9):
    """Tiny shim — using a class so tests don't need pytest.approx import."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol
        def __repr__(self):
            return f"approx({value})"
    return _Approx()
