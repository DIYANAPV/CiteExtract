"""Tests for the claim_agent prompt-variant loader.

Phase C selected v3 of the claim prompt over baseline (88.5% vs 26.9%
SUPPORTS recall on the gold set). The variant is now config-driven via
``claim_agent.prompt_variant`` so we can roll back without code changes.
These tests pin that loader so a typo in a future config doesn't silently
serve a wrong prompt.
"""

import logging
from pathlib import Path

import pytest

from src.verification.agentic.claim_agent import load_claim_prompt

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "src/verification/agentic/prompts"


class TestLoadClaimPrompt:
    def test_baseline_when_no_variant(self):
        text = load_claim_prompt(3)
        assert text == (PROMPTS_DIR / "claim_3class.txt").read_text(encoding="utf-8")

    def test_baseline_alias_loads_baseline(self):
        # "baseline" is treated as an alias for the empty default to
        # let config files spell it out explicitly.
        assert load_claim_prompt(3, "baseline") == load_claim_prompt(3, "")

    def test_v2_loads_v2_file(self):
        text = load_claim_prompt(3, "v2")
        assert text == (PROMPTS_DIR / "claim_3class_v2.txt").read_text(encoding="utf-8")

    def test_v3_loads_v3_file(self):
        text = load_claim_prompt(3, "v3")
        assert text == (PROMPTS_DIR / "claim_3class_v3.txt").read_text(encoding="utf-8")

    def test_unknown_variant_falls_back_to_baseline_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            text = load_claim_prompt(3, "v999_does_not_exist")
        # Falls back rather than raising — a typo in config shouldn't 500.
        assert text == load_claim_prompt(3, "")
        assert any("not found" in r.message.lower() for r in caplog.records)

    def test_invalid_verdict_classes_raises(self):
        with pytest.raises(ValueError, match="verdict_classes"):
            load_claim_prompt(7)

    def test_2class_baseline_still_works(self):
        # The 2-class scheme has its own baseline; variants haven't
        # been authored for it. Pin the default still loads.
        text = load_claim_prompt(2)
        assert text == (PROMPTS_DIR / "claim_2class.txt").read_text(encoding="utf-8")

    def test_v3_prompt_includes_is_it_ok_framing(self):
        # v3's distinguishing feature is the "is it OK to cite this paper
        # for this claim?" framing. If a future edit accidentally rewrites
        # v3 with a different concept, this test surfaces it.
        text = load_claim_prompt(3, "v3").lower()
        assert "is it reasonable" in text or "appropriate citation" in text
