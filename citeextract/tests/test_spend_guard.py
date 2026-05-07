
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from citeextract.verification import spend_guard


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEND_LEDGER_DIR", str(tmp_path))
    return tmp_path


def _ledger_file(ledger_dir: Path) -> Path:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    return ledger_dir / f"{month}.json"


def test_record_accumulates_across_calls(ledger_dir):
    spend_guard.record(0.10)
    spend_guard.record(0.25)
    spend_guard.record(0.05)
    assert spend_guard.current_spend_usd() == pytest.approx(0.40)

    on_disk = json.loads(_ledger_file(ledger_dir).read_text())
    assert on_disk["calls"] == 3
    assert on_disk["total_usd"] == pytest.approx(0.40)


def test_zero_or_negative_costs_are_skipped(ledger_dir):
    spend_guard.record(0)
    spend_guard.record(-1.0)
    assert spend_guard.current_spend_usd() == 0.0
    assert not _ledger_file(ledger_dir).exists()


def test_check_budget_no_op_when_env_unset(ledger_dir, monkeypatch):
    monkeypatch.delenv("MONTHLY_BUDGET_USD", raising=False)
    spend_guard.record(999.0)
    spend_guard.check_budget()


def test_check_budget_passes_when_under(ledger_dir, monkeypatch):
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "10")
    spend_guard.record(4.99)
    spend_guard.check_budget()


def test_check_budget_raises_at_or_over_cap(ledger_dir, monkeypatch):
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "10")
    spend_guard.record(10.01)
    with pytest.raises(spend_guard.BudgetExceededError) as exc:
        spend_guard.check_budget()
    assert "$10.00" in str(exc.value)
    assert "$10.01" in str(exc.value)


def test_check_budget_raises_exactly_at_cap(ledger_dir, monkeypatch):
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "5")
    spend_guard.record(5.0)
    with pytest.raises(spend_guard.BudgetExceededError):
        spend_guard.check_budget()


def test_invalid_budget_env_disables_cap(ledger_dir, monkeypatch):
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "not-a-number")
    spend_guard.record(1000.0)
    spend_guard.check_budget()


def test_zero_or_negative_budget_disables_cap(ledger_dir, monkeypatch):
    for value in ("0", "-50"):
        monkeypatch.setenv("MONTHLY_BUDGET_USD", value)
        spend_guard.check_budget()
        spend_guard.record(1.0)
        spend_guard.check_budget()


def test_corrupt_ledger_treated_as_zero(ledger_dir, monkeypatch):
    _ledger_file(ledger_dir).write_text("{not valid json")
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "10")
    spend_guard.check_budget()
    assert spend_guard.current_spend_usd() == 0.0


def test_month_rollover_uses_separate_files(ledger_dir):
    fixed_april = datetime(2026, 4, 15, tzinfo=timezone.utc)
    fixed_may = datetime(2026, 5, 1, tzinfo=timezone.utc)

    with patch("citeextract.verification.spend_guard.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_april
        mock_dt.strftime = datetime.strftime
        spend_guard.record(7.0)

    with patch("citeextract.verification.spend_guard.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_may
        mock_dt.strftime = datetime.strftime
        assert spend_guard.current_spend_usd() == 0.0
        spend_guard.record(2.0)
        assert spend_guard.current_spend_usd() == pytest.approx(2.0)

    april_file = ledger_dir / "2026-04.json"
    may_file = ledger_dir / "2026-05.json"
    assert json.loads(april_file.read_text())["total_usd"] == pytest.approx(7.0)
    assert json.loads(may_file.read_text())["total_usd"] == pytest.approx(2.0)


def test_costtracker_records_to_ledger(ledger_dir):
    from citeextract.verification.api_clients.llm_client import CostTracker

    tracker = CostTracker()
    tracker.add(1_000_000, 500_000)
    assert spend_guard.current_spend_usd() == pytest.approx(0.45)
    assert tracker.estimated_cost_usd == pytest.approx(0.45)
