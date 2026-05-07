
import logging
from pathlib import Path

import pytest

from citeextract.verification.agentic import claim_agent
from citeextract.verification.agentic.claim_agent import load_claim_prompt

PROMPTS_DIR = Path(claim_agent.__file__).parent / "prompts"


class TestLoadClaimPrompt:
    def test_baseline_when_no_variant(self):
        text = load_claim_prompt(3)
        assert text == (PROMPTS_DIR / "claim_3class.txt").read_text(encoding="utf-8")

    def test_baseline_alias_loads_baseline(self):
        assert load_claim_prompt(3, "baseline") == load_claim_prompt(3, "")

    def test_v3_loads_v3_file(self):
        text = load_claim_prompt(3, "v3")
        assert text == (PROMPTS_DIR / "claim_3class_v3.txt").read_text(encoding="utf-8")

    def test_unknown_variant_falls_back_to_baseline_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            text = load_claim_prompt(3, "v999_does_not_exist")
        assert text == load_claim_prompt(3, "")
        assert any("not found" in r.message.lower() for r in caplog.records)

    def test_invalid_verdict_classes_raises(self):
        with pytest.raises(ValueError, match="verdict_classes"):
            load_claim_prompt(7)

    def test_2class_baseline_still_works(self):
        text = load_claim_prompt(2)
        assert text == (PROMPTS_DIR / "claim_2class.txt").read_text(encoding="utf-8")

    def test_v3_prompt_includes_is_it_ok_framing(self):
        text = load_claim_prompt(3, "v3").lower()
        assert "is it reasonable" in text or "appropriate citation" in text
