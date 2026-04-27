"""Tests for citation detection and context extraction."""

import pytest

from src.citation.detector import CitationDetector, normalize_author_name
from src.citation.context_extractor import extract_context, split_sentences


# ---- Author name normalization ----

class TestNormalization:
    def test_basic(self):
        assert normalize_author_name("Smith") == "smith"

    def test_hyphens_dots_spaces(self):
        assert normalize_author_name("J.-P. Doe") == "jpdoe"

    def test_unicode(self):
        assert normalize_author_name("Müller") == "müller"


# ---- Citation Detection ----

class TestCitationDetector:
    def setup_method(self):
        self.detector = CitationDetector()

    # --- Numbered ---

    def test_single_number(self):
        text = "As shown in previous work [5], the method works."
        results = self.detector.detect_all(text)
        assert len(results) == 1
        assert results[0].keys == ["5"]
        assert results[0].marker == "[5]"

    def test_number_range(self):
        text = "Several studies [3-5] have confirmed this."
        results = self.detector.detect_all(text)
        assert len(results) == 1
        assert set(results[0].keys) == {"3", "4", "5"}

    def test_number_list(self):
        text = "Prior work [1, 3, 7] suggests otherwise."
        results = self.detector.detect_all(text)
        assert len(results) == 1
        assert set(results[0].keys) == {"1", "3", "7"}

    # --- Author-year parenthetical ---

    def test_author_year_paren(self):
        text = "This was first shown (Smith, 2020) in a landmark study."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    def test_author_year_et_al(self):
        text = "Building on (Jones et al., 2019), we extend the approach."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("jonesetal2019" in r.keys for r in results)

    def test_author_year_ampersand_paren(self):
        text = "This was shown (Smith & Jones, 2020) in a study."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    def test_three_author_apa(self):
        text = "As shown (Smith, Jones, & Lee, 2020) in their work."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    # --- Narrative ---

    def test_narrative(self):
        text = "Smith (2020) demonstrated that this works well in practice."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    def test_narrative_et_al(self):
        text = "Brown et al. (2020) proposed a new approach to few-shot learning."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("brownetal2020" in r.keys for r in results)

    def test_narrative_ampersand(self):
        text = "Smith & Jones (2020) proposed a new method."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    # --- Multiple ---

    def test_multiple_semicolon(self):
        text = "Several works (Smith, 2020; Jones, 2019) have explored this."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        keys_flat = []
        for r in results:
            keys_flat.extend(r.keys)
        assert "smith2020" in keys_flat
        assert "jones2019" in keys_flat

    # --- Harvard no-comma ---

    def test_harvard_basic(self):
        text = "As demonstrated (Smith 2020) in the literature."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    def test_harvard_et_al(self):
        text = "Previous work (Jones et al. 2019) showed improvements."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("jonesetal2019" in r.keys for r in results)

    def test_harvard_two_authors(self):
        text = "As shown (Smith and Jones 2020) the results hold."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    # --- Adjacent brackets ---

    def test_adjacent_brackets(self):
        text = "Several methods [1][2][3] have been proposed."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        keys_flat = []
        for r in results:
            keys_flat.extend(r.keys)
        assert "1" in keys_flat
        assert "2" in keys_flat
        assert "3" in keys_flat

    def test_adjacent_brackets_with_space(self):
        text = "The approach [5] [6] was tested."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        keys_flat = []
        for r in results:
            keys_flat.extend(r.keys)
        assert "5" in keys_flat
        assert "6" in keys_flat

    # --- Multi-year ---

    def test_multi_year(self):
        text = "Earlier work (Smith, 2019, 2020) confirmed the hypothesis."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        keys_flat = []
        for r in results:
            keys_flat.extend(r.keys)
        assert "smith2019" in keys_flat
        assert "smith2020" in keys_flat

    def test_multi_year_suffix(self):
        text = "As shown (Smith, 2020a, 2020b) the results differ."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        keys_flat = []
        for r in results:
            keys_flat.extend(r.keys)
        assert "smith2020a" in keys_flat
        assert "smith2020b" in keys_flat

    # --- Position tracking ---

    def test_position(self):
        text = "First sentence. The key finding [3] was important. Last sentence."
        results = self.detector.detect_all(text)
        assert len(results) == 1
        pos = results[0].position
        assert text[pos:pos + 3] == "[3]"

    # --- No citations ---

    def test_no_citations(self):
        text = "This is a plain sentence with no citations at all."
        results = self.detector.detect_all(text)
        assert len(results) == 0

    # --- False positive filtering ---

    def test_skip_table_reference(self):
        text = "As shown in Table [5], the values increase."
        results = self.detector.detect_all(text)
        assert len(results) == 0

    def test_skip_figure_reference(self):
        text = "See Figure [3] for the visualization."
        results = self.detector.detect_all(text)
        assert len(results) == 0

    def test_skip_enumeration_start_of_line(self):
        text = "[1] First item in the list\n[2] Second item\n[3] Third item"
        results = self.detector.detect_all(text)
        assert len(results) == 0

    def test_skip_editorial_sic(self):
        text = "The authro [sic] made an error."
        results = self.detector.detect_all(text)
        assert len(results) == 0

    def test_skip_math_context(self):
        text = "The formula $x = A[1, 2]$ defines the matrix."
        results = self.detector.detect_all(text)
        assert len(results) == 0

    def test_skip_math_operator_prefix(self):
        text = "The result equals = [1, 2] in the equation."
        results = self.detector.detect_all(text)
        assert len(results) == 0

    def test_skip_organization_narrative(self):
        text = "The Foundation (2020) published a report on climate change."
        results = self.detector.detect_all(text)
        assert len(results) == 0

    def test_keep_real_citation_after_organization_filter(self):
        text = "Smith (2020) published a report on climate change."
        results = self.detector.detect_all(text)
        assert len(results) >= 1
        assert any("smith2020" in r.keys for r in results)

    # --- Deduplication ---

    def test_no_duplicate_positions(self):
        text = "The study (Smith, 2020) was replicated by Jones (2019)."
        results = self.detector.detect_all(text)
        positions = [r.position for r in results]
        assert len(positions) == len(set(positions))

    # --- Mixed styles in one text ---

    def test_mixed_styles(self):
        text = (
            "Smith (2020) first showed this. "
            "Later work [5] confirmed it. "
            "Others (Jones et al., 2019) extended the results."
        )
        results = self.detector.detect_all(text)
        assert len(results) >= 3
        keys_flat = []
        for r in results:
            keys_flat.extend(r.keys)
        assert "smith2020" in keys_flat
        assert "5" in keys_flat
        assert "jonesetal2019" in keys_flat


# ---- Context Extraction ----

class TestContextExtraction:
    def test_middle_sentence(self):
        text = "First sentence here. The method [1] was tested. Third sentence follows."
        ctx = extract_context(text, text.index("[1]"), method="simple")
        assert "method" in ctx["citing_sentence"] or "[1]" in ctx["citing_sentence"]
        assert len(ctx["context_before"]) > 0
        assert len(ctx["context_after"]) > 0

    def test_first_sentence(self):
        text = "The method [1] was tested. Second sentence. Third sentence."
        ctx = extract_context(text, text.index("[1]"), method="simple")
        assert ctx["context_before"] == ""
        assert len(ctx["context_after"]) > 0

    def test_last_sentence(self):
        text = "First sentence. Second sentence. The method [1] was tested."
        ctx = extract_context(text, text.index("[1]"), method="simple")
        assert len(ctx["context_before"]) > 0
        assert ctx["context_after"] == ""

    def test_paragraph_boundary(self):
        text = (
            "Paragraph one sentence one. Paragraph one sentence two.\n\n"
            "The method [1] was tested. Second sentence of paragraph two.\n\n"
            "Paragraph three starts here."
        )
        ctx = extract_context(text, text.index("[1]"), method="simple")
        assert "[1]" in ctx["citing_sentence"]
        # Context should NOT include text from paragraph one or three
        assert "Paragraph one" not in ctx["context_before"]
        assert "Paragraph three" not in ctx["context_after"]

    def test_empty_text(self):
        ctx = extract_context("", 0, method="simple")
        assert ctx["citing_sentence"] == ""

