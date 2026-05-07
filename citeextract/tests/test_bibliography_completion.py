
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from citeextract.models.reference import Reference
from citeextract.parsers.bibliography_completion import (
    _BATCH_TARGET_CHARS,
    _SINGLE_CALL_CHAR_LIMIT,
    _surname_norm,
    _validate_and_dedup,
    assign_ref_ids,
    chunk_section,
    complete_bibliography_sync,
    extract_references_section,
    should_complete,
)


def _ref(ref_id: str, surname: str, year: int) -> Reference:
    return Reference(
        ref_id=ref_id, title=f"{surname} et al. paper",
        authors=[f"{surname} A"], year=year, source_format="grobid",
    )


def _refs(*pairs: tuple[str, str, int]) -> dict[str, Reference]:
    return {ref_id: _ref(ref_id, surname, year) for ref_id, surname, year in pairs}


def _make_pdf(tmp_path, body: str, name: str = "src.pdf") -> str:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), body, fontsize=10)
    out = tmp_path / name
    doc.save(str(out))
    doc.close()
    return str(out)


class TestShouldComplete:
    def test_few_drops_does_not_trigger(self):
        assert should_complete(2, 1, n_references=20) is False

    def test_many_drops_triggers(self):
        assert should_complete(25, 36, n_references=22) is True

    def test_drops_below_ratio_threshold_skipped(self):
        assert should_complete(3, 3, n_references=200) is False

    def test_no_existing_refs_still_triggers_on_absolute_count(self):
        assert should_complete(8, 0, n_references=0) is True


class TestChunkSection:

    def test_short_section_passes_through_as_single_chunk(self):
        section = "Smith. Title. 2020."
        assert chunk_section(section) == [section]

    def test_at_threshold_still_single_chunk(self):
        section = "x" * (_SINGLE_CALL_CHAR_LIMIT - 10)
        assert chunk_section(section) == [section]

    @staticmethod
    def _build_section(n_refs: int, line_len: int = 60) -> str:
        lines = []
        for i in range(n_refs):
            base = f"Author{i:04d}. Title text "
            pad = "x" * max(0, line_len - len(base) - 8)
            lines.append(f"{base}{pad}. 2024.")
        return "\n".join(lines)

    def test_long_section_splits_into_multiple_chunks(self):
        section = self._build_section(n_refs=1000)
        assert len(section) > _SINGLE_CALL_CHAR_LIMIT

        chunks = chunk_section(section)

        assert len(chunks) >= 2, "Long section must produce >1 chunk"
        for c in chunks:
            assert len(c) < 2 * _BATCH_TARGET_CHARS, (
                f"Chunk of {len(c)} chars overshoots {_BATCH_TARGET_CHARS} target"
            )

    def test_chunks_overlap_so_boundary_refs_are_covered(self):
        section = self._build_section(n_refs=1000)
        chunks = chunk_section(section)
        assert len(chunks) >= 2, "Need >1 chunk to test overlap"

        chunk1_tail_lines = chunks[0].split("\n")[-3:]
        chunk2_head = chunks[1]
        for line in chunk1_tail_lines:
            assert line in chunk2_head, (
                f"Boundary line {line!r} from end of chunk 1 must appear "
                "in chunk 2 to ensure cross-cut refs are fully visible "
                "to at least one batch."
            )

    def test_chunks_cover_full_section(self):
        section = self._build_section(n_refs=800)
        original_lines = section.split("\n")

        chunks = chunk_section(section)

        seen_lines: set[str] = set()
        for c in chunks:
            seen_lines.update(c.split("\n"))
        for line in original_lines:
            assert line in seen_lines, f"Line {line!r} dropped by chunker"


class TestExtractReferencesSection:
    def test_finds_section_after_heading(self, tmp_path):
        pdf = _make_pdf(
            tmp_path,
            "Body content here.\n\nReferences\n\n"
            "[1] Smith A. Title one. 2020.\n[2] Jones B. Title two. 2021.",
        )
        section = extract_references_section(pdf)
        assert "Smith" in section
        assert "Jones" in section
        assert "Body content here" not in section

    def test_truncates_at_appendix_heading(self, tmp_path):
        pdf = _make_pdf(
            tmp_path,
            "References\n\n[1] Smith A. Title. 2020.\n\n"
            "A. Limitations\nThis section discusses limits.",
        )
        section = extract_references_section(pdf)
        assert "Smith" in section
        assert "Limitations" not in section

    def test_returns_empty_when_no_heading(self, tmp_path):
        pdf = _make_pdf(tmp_path, "Body without references heading at all.")
        assert extract_references_section(pdf) == ""

    def test_author_initials_do_not_trigger_appendix_truncation(self, tmp_path):
        body = (
            "References\n\n"
            "A. M. Bran. First paper. 2023.\n"
            "M. P. Chitale. Second paper. 2025.\n"
            "S. Dhuliawala, M. Komeili, J. Xu, R. Raileanu. "
            "Third paper. 2024.\n"
            "X. Gao. Fourth paper. 2025a.\n"
            "Z. Author. Fifth paper. 2024.\n\n"
            "A. Limitations\nReal appendix body."
        )
        pdf = _make_pdf(tmp_path, body)
        section = extract_references_section(pdf)
        assert "Bran" in section
        assert "Chitale" in section
        assert "Dhuliawala" in section
        assert "Gao" in section
        assert "Z. Author" in section
        assert "Real appendix body" not in section

    def test_takes_last_references_match(self, tmp_path):
        pdf = _make_pdf(
            tmp_path,
            "We discuss References in detail below.\n\n"
            "References\n\n[1] Real bib entry. 2020.",
        )
        section = extract_references_section(pdf)
        assert "Real bib entry" in section
        assert "We discuss" not in section


class TestSurnameNorm:
    def test_simple_initial_surname(self):
        assert _surname_norm("V. Sanh") == "sanh"

    def test_lastname_first_with_initial(self):
        assert _surname_norm("Sanh, V.") == "sanh"

    def test_multiple_initials(self):
        assert _surname_norm("M. P. Chitale") == "chitale"

    def test_empty_returns_empty(self):
        assert _surname_norm("") == ""

    def test_handles_only_initials(self):
        result = _surname_norm("A B")
        assert result in {"a", "b"}


class TestValidateAndDedup:
    def test_drops_ref_with_raw_text_not_in_section(self):
        section = "Smith A. Title. 2020. Jones B. Title two. 2021."
        existing = _refs(("1", "Smith", 2020))
        raw = [{
            "title": "Hallucinated paper",
            "authors": ["Ghost X"],
            "year": 2024,
            "raw_text": "This text never appeared in the section",
        }]

        out = _validate_and_dedup(raw, section, existing)

        assert out == []

    def test_keeps_ref_whose_raw_text_appears_in_section(self):
        section = "Smith A. Title. 2020. Yamada Y. Workshop paper. 2025."
        existing = _refs(("1", "Smith", 2020))
        raw = [{
            "title": "Workshop paper",
            "authors": ["Yamada Y"],
            "year": 2025,
            "raw_text": "Yamada Y. Workshop paper. 2025.",
        }]

        out = _validate_and_dedup(raw, section, existing)

        assert len(out) == 1
        assert out[0].title == "Workshop paper"
        assert out[0].source_format == "llm_completion"
        assert out[0].ref_id == ""

    def test_dedups_against_existing_by_author_year(self):
        section = "Smith A. Title. 2020. More content here."
        existing = _refs(("1", "Smith", 2020))
        raw = [{
            "title": "Title",
            "authors": ["Smith A"],
            "year": 2020,
            "raw_text": "Smith A. Title. 2020.",
        }]

        out = _validate_and_dedup(raw, section, existing)

        assert out == []

    def test_same_year_same_author_with_suffixes_both_kept(self):
        section = (
            "X. Gao. Mmreview: A multidisciplinary benchmark. 2025a.\n"
            "X. Gao. Reviewagents: Bridging the gap. 2025b.\n"
        )
        existing: dict[str, Reference] = {}
        raw = [
            {
                "title": "Mmreview",
                "authors": ["X. Gao"],
                "year": 2025,
                "raw_text": "X. Gao. Mmreview: A multidisciplinary benchmark. 2025a.",
            },
            {
                "title": "Reviewagents",
                "authors": ["X. Gao"],
                "year": 2025,
                "raw_text": "X. Gao. Reviewagents: Bridging the gap. 2025b.",
            },
        ]

        out = _validate_and_dedup(raw, section, existing)

        titles = sorted(r.title for r in out)
        assert titles == ["Mmreview", "Reviewagents"], (
            f"Expected both 2025a and 2025b refs to be kept, got {titles}"
        )

    def test_existing_bare_year_still_blocks_llm_suffixed_duplicate(self):
        section = "X. Gao. Some paper. 2025a."
        existing = {
            "1": Reference(
                ref_id="1", title="Some paper", authors=["X. Gao"],
                year=2025, raw_text="Gao X. Some paper. 2025.",
                source_format="grobid",
            )
        }
        raw = [{
            "title": "Some paper",
            "authors": ["X. Gao"],
            "year": 2025,
            "raw_text": "X. Gao. Some paper. 2025a.",
        }]

        out = _validate_and_dedup(raw, section, existing)

        assert out == [], (
            "Bare-year existing key should match suffixed incoming key "
            "(prefix-style match) to avoid silent duplicates."
        )

    def test_drops_entry_missing_required_fields(self):
        section = "Anything"
        existing: dict[str, Reference] = {}
        raw = [{"authors": ["A"], "year": 2020, "raw_text": "Anything"}]
        assert _validate_and_dedup(raw, section, existing) == []
        raw = [{"title": "T", "year": 2020, "raw_text": "Anything"}]
        assert _validate_and_dedup(raw, section, existing) == []
        raw = [{"title": "T", "authors": ["A"], "year": 2020}]
        assert _validate_and_dedup(raw, section, existing) == []


class TestAssignRefIds:
    def test_continues_numeric_sequence(self):
        existing = _refs(("1", "Smith", 2020), ("22", "Jones", 2021))
        extras = [
            Reference(ref_id="", title="A", authors=["X"], year=2022,
                      source_format="llm_completion"),
            Reference(ref_id="", title="B", authors=["Y"], year=2023,
                      source_format="llm_completion"),
        ]

        assign_ref_ids(extras, existing)

        assert [r.ref_id for r in extras] == ["23", "24"]

    def test_falls_back_to_prefix_for_non_numeric_ids(self):
        existing = {
            "smith2020": _ref("smith2020", "Smith", 2020),
        }
        extras = [
            Reference(ref_id="", title="A", authors=["X"], year=2022,
                      source_format="llm_completion"),
        ]

        assign_ref_ids(extras, existing)

        assert extras[0].ref_id == "completed_1"


class TestCompleteBibliographySync:
    def test_full_round_trip_with_mocked_llm(self, tmp_path):
        pdf = _make_pdf(
            tmp_path,
            "Body content.\n\nReferences\n\n"
            "Smith A. Title one. 2020.\n"
            "Yamada Y. Workshop on AI. 2025.\n"
            "Jones B. Title two. 2021.",
        )
        existing = _refs(
            ("1", "Smith", 2020),
            ("2", "Jones", 2021),
        )

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = json.dumps({
            "extra_references": [
                {
                    "title": "Workshop on AI",
                    "authors": ["Yamada Y"],
                    "year": 2025,
                    "raw_text": "Yamada Y. Workshop on AI. 2025.",
                },
                {
                    "title": "Title one",
                    "authors": ["Smith A"],
                    "year": 2020,
                    "raw_text": "Smith A. Title one. 2020.",
                },
            ]
        })
        fake_response.usage = MagicMock(prompt_tokens=500, completion_tokens=200)

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch("openai.OpenAI", return_value=fake_client):
            extra, cost = complete_bibliography_sync(
                pdf, existing, llm_config={"model": "gpt-4o-mini"},
                api_key="sk-fake",
            )

        assert len(extra) == 1
        assert extra[0].title == "Workshop on AI"
        assert extra[0].source_format == "llm_completion"
        assert cost > 0

    def test_no_section_returns_empty_at_zero_cost(self, tmp_path):
        pdf = _make_pdf(tmp_path, "Body with no bibliography heading.")

        with patch("openai.OpenAI") as mock_openai:
            extra, cost = complete_bibliography_sync(
                pdf, {}, llm_config={"model": "gpt-4o-mini"},
                api_key="sk-fake",
            )

        assert extra == []
        assert cost == 0.0
        mock_openai.assert_not_called()

    def test_run_batches_makes_one_llm_call_per_chunk(self):
        from citeextract.parsers.bibliography_completion import _run_batches

        chunks = [f"chunk_{i} text content goes here" for i in range(4)]

        per_call_response = MagicMock()
        per_call_response.choices = [MagicMock()]
        per_call_response.choices[0].message.content = json.dumps({
            "references": [{"title": "T", "authors": ["A"], "year": 2024,
                            "raw_text": "raw"}]
        })
        per_call_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = per_call_response

        with patch("openai.OpenAI", return_value=fake_client):
            raw_refs, cost = _run_batches(chunks, {"model": "gpt-4o-mini"}, "sk-fake")

        assert fake_client.chat.completions.create.call_count == 4
        assert len(raw_refs) == 4
        assert cost > 0

    def test_run_batches_single_chunk_skips_threadpool(self):
        from citeextract.parsers.bibliography_completion import _run_batches

        per_call_response = MagicMock()
        per_call_response.choices = [MagicMock()]
        per_call_response.choices[0].message.content = json.dumps({
            "references": [{"title": "T", "authors": ["A"], "year": 2024,
                            "raw_text": "raw"}]
        })
        per_call_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = per_call_response

        with patch("openai.OpenAI", return_value=fake_client):
            raw_refs, _ = _run_batches(["only chunk"], {"model": "gpt-4o-mini"}, "sk-fake")

        assert fake_client.chat.completions.create.call_count == 1
        assert len(raw_refs) == 1

    def test_invalid_json_response_returns_empty_silently(self, tmp_path):
        pdf = _make_pdf(
            tmp_path,
            "References\n\nSmith A. Title. 2020. More text here for length.",
        )

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "{not valid json"
        fake_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch("openai.OpenAI", return_value=fake_client):
            extra, cost = complete_bibliography_sync(
                pdf, {}, llm_config={"model": "gpt-4o-mini"},
                api_key="sk-fake",
            )

        assert extra == []
        assert cost == 0.0
