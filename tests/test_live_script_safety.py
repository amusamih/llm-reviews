from __future__ import annotations

from types import SimpleNamespace

import evaluation.live_baseline_ablation as live_baseline_ablation
import evaluation.live_model_substitution as live_model_substitution
import evaluation.live_pilot as live_pilot


def test_live_pilot_requires_both_cli_flags(monkeypatch, capsys):
    calls: list[str] = []
    monkeypatch.setenv("ALLOW_LIVE_LLM", "true")
    monkeypatch.setattr(live_pilot, "run_live_pilot", lambda **kwargs: calls.append("called"))

    monkeypatch.setattr("sys.argv", ["live_pilot.py"])
    live_pilot.main()
    monkeypatch.setattr("sys.argv", ["live_pilot.py", "--run-live"])
    live_pilot.main()
    monkeypatch.setattr("sys.argv", ["live_pilot.py", "--allow-live"])
    live_pilot.main()

    assert calls == []
    assert "Re-run with both --run-live and --allow-live" in capsys.readouterr().out


def test_live_pilot_runs_only_after_two_cli_flags(monkeypatch, capsys):
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return _artifacts()

    monkeypatch.setattr(live_pilot, "run_live_pilot", fake_run)
    monkeypatch.setattr("sys.argv", ["live_pilot.py", "--run-live", "--allow-live", "--max-prompts", "1"])

    live_pilot.main()

    assert len(calls) == 1
    assert calls[0]["max_prompts"] == 1
    assert "run_id=unit-run" in capsys.readouterr().out


def test_live_baseline_ablation_requires_both_cli_flags(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("ALLOW_LIVE_LLM", "true")
    monkeypatch.setattr(live_baseline_ablation, "run_live_baseline_ablation", lambda **kwargs: calls.append("called"))

    for argv in (
        ["live_baseline_ablation.py"],
        ["live_baseline_ablation.py", "--run-live"],
        ["live_baseline_ablation.py", "--allow-live"],
    ):
        monkeypatch.setattr("sys.argv", argv)
        live_baseline_ablation.main()

    assert calls == []


def test_live_baseline_ablation_runs_only_after_two_cli_flags(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        live_baseline_ablation,
        "run_live_baseline_ablation",
        lambda **kwargs: calls.append(kwargs) or _artifacts(),
    )
    monkeypatch.setattr("sys.argv", ["live_baseline_ablation.py", "--run-live", "--allow-live", "--max-prompts", "1"])

    live_baseline_ablation.main()

    assert len(calls) == 1
    assert calls[0]["max_prompts"] == 1


def test_live_model_substitution_requires_both_cli_flags(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("ALLOW_LIVE_LLM", "true")
    monkeypatch.setattr(live_model_substitution, "run_live_model_substitution", lambda **kwargs: calls.append("called"))

    for argv in (
        ["live_model_substitution.py"],
        ["live_model_substitution.py", "--run-live"],
        ["live_model_substitution.py", "--allow-live"],
    ):
        monkeypatch.setattr("sys.argv", argv)
        live_model_substitution.main()

    assert calls == []


def test_live_model_substitution_runs_only_after_two_cli_flags(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        live_model_substitution,
        "run_live_model_substitution",
        lambda **kwargs: calls.append(kwargs) or _artifacts(),
    )
    monkeypatch.setattr("sys.argv", ["live_model_substitution.py", "--run-live", "--allow-live", "--max-prompts", "1"])

    live_model_substitution.main()

    assert len(calls) == 1
    assert calls[0]["max_prompts"] == 1


def _artifacts():
    return SimpleNamespace(
        run_id="unit-run",
        manifest_path="manifest.json",
        results_path="results.jsonl",
        summary_path="summary.json",
        evidence_path="evidence.json",
    )
