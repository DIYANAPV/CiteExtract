
import pytest

from citeextract.verification.title_comparison import compare_titles


class TestExactMatch:
    def test_identical_titles(self):
        result = compare_titles(
            "Attention Is All You Need",
            "Attention Is All You Need",
        )
        assert result.exact_match is True
        assert result.differences == []
        assert result.similarity == 1.0

    def test_case_insensitive(self):
        result = compare_titles(
            "attention is all you need",
            "Attention Is All You Need",
        )
        assert result.exact_match is True

    def test_unicode_normalization(self):
        result = compare_titles(
            "Über die Grundlagen",
            "Uber die Grundlagen",
        )
        assert result.exact_match is True

    def test_punctuation_ignored(self):
        result = compare_titles(
            "Attention Is All You Need.",
            "Attention Is All You Need",
        )
        assert result.exact_match is True


class TestDifferences:
    def test_one_word_different(self):
        result = compare_titles(
            "Attention Is All You Need",
            "Attention Is All We Need",
        )
        assert result.exact_match is False
        assert len(result.differences) >= 1
        assert any("you" in d and "we" in d for d in result.differences)

    def test_extra_word_in_ref(self):
        result = compare_titles(
            "A Novel Deep Learning Approach",
            "A Deep Learning Approach",
        )
        assert result.exact_match is False
        assert any("novel" in d.lower() for d in result.differences)

    def test_extra_word_in_db(self):
        result = compare_titles(
            "Deep Learning Approach",
            "A Deep Learning Approach",
        )
        assert result.exact_match is False
        assert any("extra" in d.lower() or "db has" in d.lower() for d in result.differences)

    def test_completely_different(self):
        result = compare_titles(
            "Attention Is All You Need",
            "Deep Residual Learning for Image Recognition",
        )
        assert result.exact_match is False
        assert len(result.differences) >= 1
        assert result.similarity < 0.5


class TestEdgeCases:
    def test_empty_ref(self):
        result = compare_titles("", "Some Title")
        assert result.exact_match is False
        assert result.similarity == 0.0

    def test_empty_db(self):
        result = compare_titles("Some Title", "")
        assert result.exact_match is False
        assert result.similarity == 0.0

    def test_both_empty(self):
        result = compare_titles("", "")
        assert result.exact_match is False

    def test_version_marker_stripped(self):
        result = compare_titles(
            "Language Models v2",
            "Language Models",
        )
        assert result.exact_match is True


class TestSerialization:
    def test_to_dict_exact(self):
        result = compare_titles("Same Title", "Same Title")
        d = result.to_dict()
        assert d["exact_match"] is True
        assert "differences" not in d

    def test_to_dict_with_differences(self):
        result = compare_titles("Title A", "Title B")
        d = result.to_dict()
        assert d["exact_match"] is False
        assert "differences" in d
        assert isinstance(d["differences"], list)
        assert isinstance(d["similarity"], float)
