"""Tests for citation format detection."""

import pytest

from src.citation.format_detector import (
    detect_citation_format,
    get_format_rules,
    FORMAT_RULES,
)


# ---------------------------------------------------------------------------
# APA format examples
# ---------------------------------------------------------------------------

class TestAPADetection:
    """APA: Surname, A. B., & Surname, C. D. (Year). Title. Journal, Vol(Issue), pages."""

    def test_apa_journal_article(self):
        raw = (
            'Smith, J. A., & Jones, B. C. (2020). The effects of climate change '
            'on biodiversity. Nature, 580(7804), 396-401.'
        )
        assert detect_citation_format(raw) == "apa"

    def test_apa_multiple_authors(self):
        raw = (
            'Brown, T. B., Mann, B., Ryder, N., Subbiah, M., & Kaplan, J. (2020). '
            'Language models are few-shot learners. Advances in Neural Information '
            'Processing Systems, 33, 1877-1901.'
        )
        assert detect_citation_format(raw) == "apa"

    def test_apa_with_doi(self):
        raw = (
            'Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., '
            'Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. '
            'Advances in Neural Information Processing Systems, 30. '
            'https://doi.org/10.48550/arXiv.1706.03762'
        )
        assert detect_citation_format(raw) == "apa"

    def test_apa_single_author(self):
        raw = (
            'Chomsky, N. (1957). Syntactic structures. Mouton & Co.'
        )
        assert detect_citation_format(raw) == "apa"


# ---------------------------------------------------------------------------
# Vancouver format examples
# ---------------------------------------------------------------------------

class TestVancouverDetection:
    """Vancouver: Surname AB, Surname CD. Title. Journal. Year;Vol(Issue):pages."""

    def test_vancouver_journal(self):
        raw = (
            'Huang H, LeCun Y, Balestriero R. Llm-jepa: Large language models '
            'meet joint embedding predictive architectures. arXiv preprint '
            'arXiv:2509.14252. 2025.'
        )
        assert detect_citation_format(raw) == "vancouver"

    def test_vancouver_with_volume(self):
        raw = (
            'Smith AB, Jones CD, Lee EF. A novel method for protein folding. '
            'J Mol Biol. 2019;381(4):431-445.'
        )
        assert detect_citation_format(raw) == "vancouver"

    def test_vancouver_et_al(self):
        raw = (
            'Wilson AB, Thomas CD, Brown EF, et al. Advances in gene therapy. '
            'Nat Med. 2021;27(3):512-520.'
        )
        assert detect_citation_format(raw) == "vancouver"


# ---------------------------------------------------------------------------
# IEEE format examples
# ---------------------------------------------------------------------------

class TestIEEEDetection:
    """IEEE: A. B. Surname, C. D. Surname, "Title," in Proc. Conf, Year, pp. X-Y."""

    def test_ieee_conference(self):
        raw = (
            'A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification '
            'with deep convolutional neural networks," in Proc. Advances in Neural '
            'Information Processing Systems, 2012, pp. 1097-1105.'
        )
        assert detect_citation_format(raw) == "ieee"

    def test_ieee_journal(self):
        raw = (
            'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for '
            'image recognition," in Proc. IEEE Conf. Computer Vision and Pattern '
            'Recognition, 2016, pp. 770-778.'
        )
        assert detect_citation_format(raw) == "ieee"

    def test_ieee_single_initial(self):
        raw = (
            'Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based '
            'learning applied to document recognition," in Proc. IEEE, vol. 86, '
            'no. 11, 1998, pp. 2278-2324.'
        )
        assert detect_citation_format(raw) == "ieee"


# ---------------------------------------------------------------------------
# Chicago format examples
# ---------------------------------------------------------------------------

class TestChicagoDetection:
    """Chicago: Surname, Firstname. "Title." Journal Vol, no. Issue (Year): pages."""

    def test_chicago_article(self):
        raw = (
            'Smith, Jonathan. "The Role of Neural Networks in Modern AI." '
            'Journal of Artificial Intelligence 45, no. 3 (2020): 112-130.'
        )
        assert detect_citation_format(raw) == "chicago"

    def test_chicago_book(self):
        raw = (
            'Goodfellow, Ian. "Deep Learning." '
            'Journal of AI Research 12, no. 4 (2016): 1-50.'
        )
        assert detect_citation_format(raw) == "chicago"


# ---------------------------------------------------------------------------
# Harvard format examples
# ---------------------------------------------------------------------------

class TestHarvardDetection:
    """Harvard: Surname, A.B. Year, 'Title', Journal, vol. X, no. Y, pp. Z."""

    def test_harvard_journal(self):
        raw = (
            "Smith, A.B. 2020, 'A novel approach to deep learning', "
            "Journal of Machine Learning, vol. 12, no. 3, pp. 45-67."
        )
        assert detect_citation_format(raw) == "harvard"

    def test_harvard_with_multiple_keywords(self):
        raw = (
            "Brown, T.B. 2019, 'Language models and their applications', "
            "Computational Linguistics, vol. 45, no. 2, pp. 201-234."
        )
        assert detect_citation_format(raw) == "harvard"


# ---------------------------------------------------------------------------
# MLA format examples
# ---------------------------------------------------------------------------

class TestMLADetection:
    """MLA: Surname, Firstname. "Title." Journal, vol. X, no. Y, Year, pp. Z."""

    def test_mla_article(self):
        raw = (
            'Smith, Jonathan. "The Future of Artificial Intelligence." '
            'Journal of Computer Science, vol. 15, no. 2, 2020, pp. 45-67.'
        )
        assert detect_citation_format(raw) == "mla"


# ---------------------------------------------------------------------------
# BibTeX format
# ---------------------------------------------------------------------------

class TestBibTeXDetection:
    """BibTeX: detected via source_format, not raw_text patterns."""

    def test_bibtex_via_source_format(self):
        raw = "Brown, Tom B et al. (2020). Language models are few-shot learners. NeurIPS"
        assert detect_citation_format(raw, source_format="bibtex") == "bibtex"

    def test_bibtex_overrides_content(self):
        # Even if raw_text looks like APA, bibtex source_format wins
        raw = (
            'Smith, J. A., & Jones, B. C. (2020). Title. Journal, 580(7804), 396-401.'
        )
        assert detect_citation_format(raw, source_format="bibtex") == "bibtex"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        assert detect_citation_format("") is None

    def test_too_short(self):
        assert detect_citation_format("Smith 2020") is None

    def test_none_like_input(self):
        assert detect_citation_format("   ") is None

    def test_ambiguous_returns_none(self):
        # A very generic reference that could be multiple formats
        raw = "Author. Title. 2020."
        assert detect_citation_format(raw) is None


# ---------------------------------------------------------------------------
# Format rules
# ---------------------------------------------------------------------------

class TestFormatRules:
    def test_all_formats_have_rules(self):
        for fmt in ["apa", "vancouver", "ieee", "chicago", "harvard", "mla", "bibtex"]:
            rules = get_format_rules(fmt)
            assert rules is not None
            assert rules.name

    def test_unknown_format_returns_none(self):
        assert get_format_rules(None) is None
        assert get_format_rules("unknown") is None

    def test_apa_truncation_threshold(self):
        rules = get_format_rules("apa")
        assert rules.et_al_after == 20

    def test_vancouver_truncation_threshold(self):
        rules = get_format_rules("vancouver")
        assert rules.et_al_after == 6

    def test_mla_truncation_threshold(self):
        rules = get_format_rules("mla")
        assert rules.et_al_after == 1
