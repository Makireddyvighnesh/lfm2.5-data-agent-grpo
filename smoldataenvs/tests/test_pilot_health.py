import gzip
import json

from pirl.artifacts import append_jsonl, save_rollout, write_json
from pirl.run_health import validate_run


def make_run(run, steps=2):
    for step in range(1, steps + 1):
        row = {"rollout_id": f"r{step}", "step": step, "reward": 1.0, "correct": 1.0, "sequences": []}
        trace = {"turns": [{"prompt_token_ids": [10], "completion_token_ids": [20],
                           "per_token_logps": [-0.2], "loss_mask": [0, 1]}]}
        save_rollout(run, row, {"name": "task", "instruction": "q", "difficulty": "hard"}, {}, trace)
        append_jsonl(run / "rollouts.jsonl", {k: v for k, v in row.items() if k != "sequences"})
        append_jsonl(run / "metrics.jsonl", {"step": step, "rollouts": 1, "capture_errors": 0,
                     "infra_errors": 0, "cleanup_errors": 0, "openenv_loop_threads": 0,
                     "loss": -0.1, "ratio_mean": 1.0, "kl": 0.0, "clip_frac": 0.0,
                     "grad_norm": 0.1, "trained_completion_tokens": 1, "process_fds": 12})
    write_json(run / "state_latest/state.json", {"step": steps})


def evaluations(tmp_path, candidate_accuracy=0.3, template="prefix-chain"):
    base = {"harness": "pi", "split": "test", "instruction_hash": "same", "difficulty_filter": "medium,hard",
            "template": "prefix-chain", "temperature": 1, "top_p": 1, "top_k": -1, "max_tokens": 4096,
            "step_limit": 17, "agent_timeout": 300, "rollouts_per_task": 1, "coverage": 1,
            "expected": 50, "pass_at_1": 0.2, "trainable_rate": 1, "atif_match_rate": 1}
    candidate = {**base, "pass_at_1": candidate_accuracy, "template": template}
    baseline_path, candidate_path = tmp_path / "base.json", tmp_path / "candidate.json"
    write_json(baseline_path, base)
    write_json(candidate_path, candidate)
    return baseline_path, candidate_path


def test_pilot_gate_accepts_complete_captures_and_matching_improvement(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    baseline, candidate = evaluations(tmp_path)
    report = validate_run(run, 2, baseline, candidate)
    assert report["passed"] and report["saved_rollouts"] == 2


def test_pilot_gate_rejects_corrupted_capture(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    path = run / "rollouts/r1.json.gz"
    with gzip.open(path, "rt") as stream:
        bundle = json.load(stream)
    bundle["training_trace"]["turns"][0]["per_token_logps"] = []
    write_json(path, bundle, compressed=True)
    report = validate_run(run, 2)
    assert not report["passed"] and len(report["capture_errors"]) == 1


def test_pilot_gate_rejects_different_templates_or_accuracy_regression(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    baseline, candidate = evaluations(tmp_path, 0.1, "native")
    report = validate_run(run, 2, baseline, candidate)
    assert not report["passed"]
    assert any("settings/tasks differ" in issue for issue in report["issues"])
    assert any("not improved" in issue for issue in report["issues"])


def test_pilot_gate_rejects_graded_evaluation_with_broken_capture(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    baseline, candidate = evaluations(tmp_path)
    trained = json.loads(candidate.read_text())
    trained.update(trainable_rate=0.98, atif_match_rate=0.98)
    write_json(candidate, trained)
    report = validate_run(run, 2, baseline, candidate)
    assert not report["passed"]
    assert any("candidate evaluation" in issue and "token-capture" in issue for issue in report["issues"])


def test_pipeline_never_extends_when_pilot_health_fails(tmp_path, monkeypatch):
    from pirl import run_pilot
    import pytest

    commands = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pilot, "ROOT", tmp_path)
    monkeypatch.setattr(run_pilot.sys, "argv", ["run_pilot", "--output", str(tmp_path / "runs/pilot")])
    monkeypatch.setattr(run_pilot, "execute", lambda argv, env: commands.append(list(map(str, argv))))
    monkeypatch.setattr(run_pilot, "validate_run", lambda *a: {"passed": False, "issues": ["leaked loops"]})
    with pytest.raises(SystemExit, match="health checks failed"):
        run_pilot.main()
    targets = [cmd[cmd.index("--steps") + 1] for cmd in commands if "--steps" in cmd]
    assert targets == ["50"]


def test_pipeline_extends_same_run_only_after_passing_evaluation_gate(tmp_path, monkeypatch):
    from pirl import run_pilot

    commands = []
    gates = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pilot, "ROOT", tmp_path)
    monkeypatch.setattr(run_pilot.sys, "argv", ["run_pilot", "--output", str(tmp_path / "runs/pilot")])
    monkeypatch.setattr(run_pilot, "execute", lambda argv, env: commands.append(list(map(str, argv))))

    def validate(*args):
        gates.append(args)
        return {"passed": True, "issues": []}

    monkeypatch.setattr(run_pilot, "validate_run", validate)
    run_pilot.main()
    training = [cmd for cmd in commands if "--steps" in cmd]
    assert [cmd[cmd.index("--steps") + 1] for cmd in training] == ["50", "200"]
    assert training[0][2] == training[1][2]  # extend the saved policy, not a second fresh run
    assert len(gates[1]) == 4  # second gate compares both evaluation summaries
