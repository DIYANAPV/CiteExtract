"""Tests for all parsers and the format router."""

import os
from pathlib import Path

import pytest

from src.models.parsed_paper import ParsedPaper
from src.parsers.bibtex_parser import BibtexParser
from src.parsers.latex_parser import LatexParser
from src.parsers.text_parser import TextParser
from src.parsers.router import parse_file, UnsupportedFormatError, InputTooLargeError

FIXTURES = Path(__file__).parent / "fixtures"


# ---- BibTeX Parser ----

class TestBibtexParser:
    def test_parse_sample_bib(self):
        parser = BibtexParser()
        result = parser.parse(str(FIXTURES / "sample.bib"))

        assert isinstance(result, ParsedPaper)
        assert result.input_format == "bibtex"
        assert result.has_body_text is False
        assert len(result.citations) == 0
        assert len(result.references) == 5

    def test_reference_fields(self):
        parser = BibtexParser()
        result = parser.parse(str(FIXTURES / "sample.bib"))

        ref_by_id = {r.ref_id: r for r in result.references}

        # Check Vaswani et al.
        vaswani = ref_by_id["vaswani2017attention"]
        assert vaswani.title == "Attention is All You Need"
        assert vaswani.year == 2017
        assert len(vaswani.authors) == 8
        assert vaswani.source_format == "bibtex"

        # Check BERT
        bert = ref_by_id["devlin2019bert"]
        assert "BERT" in bert.title
        assert bert.year == 2019

        # Check book
        manning = ref_by_id["manning2008introduction"]
        assert manning.year == 2008
        assert "Cambridge" in (manning.venue or "")

    def test_empty_bib(self, tmp_path):
        bib_file = tmp_path / "empty.bib"
        bib_file.write_text("")
        result = BibtexParser().parse(str(bib_file))
        assert len(result.references) == 0
        assert result.has_body_text is False

    def test_bib_warning(self):
        result = BibtexParser().parse(str(FIXTURES / "sample.bib"))
        assert any("semantic verification" in w.lower() for w in result.warnings)

    def test_can_parse(self):
        parser = BibtexParser()
        assert parser.can_parse("refs.bib") is True
        assert parser.can_parse("paper.pdf") is False


# ---- LaTeX Parser ----

class TestLatexParser:
    def test_parse_sample_tex(self):
        parser = LatexParser()
        result = parser.parse(str(FIXTURES / "sample.tex"))

        assert isinstance(result, ParsedPaper)
        assert result.input_format == "latex"
        assert result.has_body_text is True

    def test_references_from_bib(self):
        result = LatexParser().parse(str(FIXTURES / "sample.tex"))
        ref_ids = {r.ref_id for r in result.references}
        assert "smith2020deep" in ref_ids
        assert "jones2019transformers" in ref_ids
        assert "lee2021attention" in ref_ids

    def test_citations_extracted(self):
        result = LatexParser().parse(str(FIXTURES / "sample.tex"))
        assert len(result.citations) > 0
        cited_refs = {c.ref_id for c in result.citations}
        assert "smith2020deep" in cited_refs

    def test_citation_context(self):
        result = LatexParser().parse(str(FIXTURES / "sample.tex"))
        # Find a citation for smith2020deep
        smith_cites = [c for c in result.citations if c.ref_id == "smith2020deep"]
        assert len(smith_cites) > 0
        # The citing sentence should contain some text
        assert len(smith_cites[0].citing_sentence) > 10

    def test_metadata(self):
        result = LatexParser().parse(str(FIXTURES / "sample.tex"))
        assert "title" in result.metadata
        assert "Survey" in result.metadata["title"]

    def test_missing_bib(self, tmp_path):
        tex_file = tmp_path / "orphan.tex"
        tex_file.write_text(r"""
\documentclass{article}
\begin{document}
Some text with a citation \cite{foo2020}.
\end{document}
""")
        result = LatexParser().parse(str(tex_file))
        assert any("no .bib" in w.lower() for w in result.warnings)
        # Should still create placeholder references
        assert len(result.references) > 0


# ---- Text Parser ----

class TestTextParser:
    def test_parse_sample_txt(self):
        parser = TextParser()
        result = parser.parse(str(FIXTURES / "sample.txt"))

        assert isinstance(result, ParsedPaper)
        assert result.input_format == "text"
        assert result.has_body_text is True

    def test_references_extracted(self):
        result = TextParser().parse(str(FIXTURES / "sample.txt"))
        assert len(result.references) >= 2

    def test_citations_extracted(self):
        result = TextParser().parse(str(FIXTURES / "sample.txt"))
        assert len(result.citations) > 0
        # Should find numbered citations like [1], [2]
        markers = {c.marker for c in result.citations}
        assert any("[1]" in m for m in markers) or any("[2]" in m for m in markers)

    def test_citation_context_not_empty(self):
        result = TextParser().parse(str(FIXTURES / "sample.txt"))
        for cit in result.citations:
            assert len(cit.citing_sentence) > 5

    def test_no_reference_section(self, tmp_path):
        txt_file = tmp_path / "noref.txt"
        txt_file.write_text("Just some plain text without any references.")
        result = TextParser().parse(str(txt_file))
        assert len(result.references) == 0
        assert any("no reference section" in w.lower() for w in result.warnings)


# ---- Format Router ----

class TestRouter:
    def test_route_bib(self):
        result = parse_file(str(FIXTURES / "sample.bib"))
        assert result.input_format == "bibtex"

    def test_route_tex(self):
        result = parse_file(str(FIXTURES / "sample.tex"))
        assert result.input_format == "latex"

    def test_route_txt(self):
        result = parse_file(str(FIXTURES / "sample.txt"))
        assert result.input_format == "text"

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b,c")
        with pytest.raises(UnsupportedFormatError):
            parse_file(str(bad_file))

    def test_consistent_output_structure(self):
        """All parsers should produce valid ParsedPaper with same schema."""
        bib = parse_file(str(FIXTURES / "sample.bib"))
        tex = parse_file(str(FIXTURES / "sample.tex"))
        txt = parse_file(str(FIXTURES / "sample.txt"))

        for result in [bib, tex, txt]:
            assert isinstance(result, ParsedPaper)
            assert isinstance(result.references, list)
            assert isinstance(result.citations, list)
            assert isinstance(result.has_body_text, bool)
            assert isinstance(result.warnings, list)

    def test_bib_has_no_body_text(self):
        result = parse_file(str(FIXTURES / "sample.bib"))
        assert result.has_body_text is False

    def test_tex_has_body_text(self):
        result = parse_file(str(FIXTURES / "sample.tex"))
        assert result.has_body_text is True

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_file("/nonexistent/path/paper.bib")

    def test_file_too_large(self, tmp_path):
        """Files exceeding MAX_FILE_SIZE_MB should be rejected."""
        from src.parsers.router import MAX_FILE_SIZE_MB
        big_file = tmp_path / "huge.bib"
        # Write a file just over the limit
        big_file.write_bytes(b"x" * (MAX_FILE_SIZE_MB * 1024 * 1024 + 1))
        with pytest.raises(InputTooLargeError):
            parse_file(str(big_file))


# ---- GROBID parser: reference dedup + citation remap ----


class TestGrobidDedupRemap:
    """Regression guards for the citation-orphaning bug when two refs
    share a title.

    GROBID + the LLM ref parser can occasionally emit two ``Reference``
    entries with the same title (different ``raw_text`` whitespace,
    different field set, etc.). ``_deduplicate_references`` keeps one and
    drops the other. Citations that pointed at the dropped one used to
    be silently orphaned — the annotator's ``skipped_no_verdict`` counter
    would tick up and the user would see a missing comment-box.

    These tests pin down: (a) the remap is built correctly, and
    (b) the kept ref is the first occurrence (stable for cache replay).
    """

    @staticmethod
    def _ref(ref_id: str, title: str, year: int = 2020):
        from src.models.reference import Reference
        return Reference(
            ref_id=ref_id, title=title, authors=["A"],
            year=year, source_format="grobid",
        )

    def test_no_dupes_returns_empty_remap(self):
        from src.parsers.grobid_parser import GrobidParser

        refs = {
            "1": self._ref("1", "Title One"),
            "2": self._ref("2", "Title Two"),
        }
        deduped, remap = GrobidParser._deduplicate_references(refs)

        assert remap == {}
        assert set(deduped.keys()) == {"1", "2"}

    def test_dropped_ref_appears_in_remap_with_kept_target(self):
        from src.parsers.grobid_parser import GrobidParser

        # Refs 1 and 3 share a title; ref 1 is the first occurrence and
        # should be the kept survivor.
        refs = {
            "1": self._ref("1", "Shared Title"),
            "2": self._ref("2", "Other Title"),
            "3": self._ref("3", "Shared Title"),
        }
        deduped, remap = GrobidParser._deduplicate_references(refs)

        assert set(deduped.keys()) == {"1", "2"}, "Survivor should be the first-seen duplicate"
        assert remap == {"3": "1"}, "Dropped ref must map to its kept twin"

    def test_first_occurrence_wins_even_with_three_duplicates(self):
        from src.parsers.grobid_parser import GrobidParser

        refs = {
            "1": self._ref("1", "Same Title"),
            "5": self._ref("5", "Same Title"),
            "9": self._ref("9", "Same Title"),
        }
        deduped, remap = GrobidParser._deduplicate_references(refs)

        assert list(deduped.keys()) == ["1"]
        assert remap == {"5": "1", "9": "1"}

    def test_empty_title_is_not_deduped(self):
        """Refs with no title are kept as distinct — without a title we
        can't safely declare them duplicates."""
        from src.parsers.grobid_parser import GrobidParser

        refs = {
            "1": self._ref("1", ""),
            "2": self._ref("2", ""),
        }
        deduped, remap = GrobidParser._deduplicate_references(refs)

        assert set(deduped.keys()) == {"1", "2"}
        assert remap == {}
