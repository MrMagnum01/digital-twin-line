"""runner.cli: argument-level refusal and exclusive-artifact behaviour only
(dev-table happy path itself is already covered end to end by
test_runner_pipeline.py)."""
import json
import subprocess
import sys

import pytest

from runner import cli, lock_guard
from runner.cli import ArtifactExistsError, main


def test_cli_refuses_locked_table_without_the_clearance_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--table", "locked"])
    assert exc.value.code == 2
    assert "requires --i-have-clearance-to-run-locked-seeds" in capsys.readouterr().err


def test_cli_writes_exclusive_artifacts_for_the_dev_table(tmp_path):
    run_dir = tmp_path / "run1"
    main(["--table", "dev", "--run-dir", str(run_dir)])

    pre_run = json.loads((run_dir / "pre_run.json").read_text())
    assert pre_run["table"] == "dev"
    assert "runner/pipeline.py" in pre_run["runner_implementation_sha256"]

    selection = json.loads((run_dir / "selection.json").read_text())
    assert selection["lock_version"] == "lock-3"
    assert "static_threshold_selection" in selection
    assert "isolation_forest_selection" in selection

    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["table"] == "dev"
    assert evaluation["lock_version"] == "lock-3"

    assert not (run_dir / "failure.json").exists()


# --- Astra freeze-review MUST-FIX 4 (2026-09-28): exclusive, never-
# overwritten pre-run/selection/evaluation/failure artifacts. ---

def test_cli_refuses_to_overwrite_an_existing_run_dir(tmp_path):
    run_dir = tmp_path / "run1"
    main(["--table", "dev", "--run-dir", str(run_dir)])
    with pytest.raises(ArtifactExistsError):
        main(["--table", "dev", "--run-dir", str(run_dir)])
    # the second (refused) attempt must not have clobbered the first report
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["table"] == "dev"


def test_cli_records_a_failure_artifact_when_selection_raises(tmp_path, monkeypatch):
    run_dir = tmp_path / "run2"

    def _boom(table_name, *, allow_locked=False, repo_root=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.pipeline, "select", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        main(["--table", "dev", "--run-dir", str(run_dir)])

    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "boom"
    assert not (run_dir / "selection.json").exists()
    assert not (run_dir / "evaluation.json").exists()


# --- Astra freeze-review r2 group 4 (2026-09-28): selection.json must be
# written exclusively BEFORE any test-split scoring, including when test
# scoring itself fails - the validation-only selection must still survive
# on disk as a durable record. ---

def test_cli_records_a_failure_artifact_when_test_scoring_raises_but_keeps_selection(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "run3"

    def _boom(sel):
        raise RuntimeError("test scoring boom")

    monkeypatch.setattr(cli.pipeline, "score_test", _boom)
    with pytest.raises(RuntimeError, match="test scoring boom"):
        main(["--table", "dev", "--run-dir", str(run_dir)])

    assert (run_dir / "selection.json").exists()
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "test scoring boom"
    assert not (run_dir / "evaluation.json").exists()


def test_cli_writes_selection_json_before_any_test_split_scoring(tmp_path, monkeypatch):
    """The literal r2 group 4 probe: a fixture interception must observe
    selection.json already present on disk the moment test-split scoring
    begins, not merely absent from the in-memory report shape."""
    import evaluator

    run_dir = tmp_path / "run4"
    seen_before_test_scoring = {}
    real_score_stratum = evaluator.score_stratum

    def _spy(inp, events, cfg=None):
        if inp.split == "test" and "selection_json_existed" not in seen_before_test_scoring:
            seen_before_test_scoring["selection_json_existed"] = (run_dir / "selection.json").exists()
        return real_score_stratum(inp, events, cfg)

    monkeypatch.setattr(evaluator, "score_stratum", _spy)
    monkeypatch.setattr(cli.pipeline.evaluator, "score_stratum", _spy)
    main(["--table", "dev", "--run-dir", str(run_dir)])
    assert seen_before_test_scoring == {"selection_json_existed": True}


# --- Astra freeze-review r3 group 4 (2026-09-28): output-write failures
# for selection.json/evaluation.json must themselves be captured by the
# failure record, not left as a bare exception with no artifact trace. ---

def test_cli_records_a_failure_artifact_when_selection_write_raises(tmp_path, monkeypatch):
    run_dir = tmp_path / "run5"
    real_write = cli._write_exclusive

    def _flaky(path, data):
        if path.name == "selection.json":
            raise OSError("disk full")
        return real_write(path, data)

    monkeypatch.setattr(cli, "_write_exclusive", _flaky)
    with pytest.raises(OSError, match="disk full"):
        main(["--table", "dev", "--run-dir", str(run_dir)])
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["exception_type"] == "OSError"
    assert not (run_dir / "evaluation.json").exists()


def test_cli_records_a_failure_artifact_when_evaluation_write_raises(tmp_path, monkeypatch):
    run_dir = tmp_path / "run6"
    real_write = cli._write_exclusive

    def _flaky(path, data):
        if path.name == "evaluation.json":
            raise OSError("disk full")
        return real_write(path, data)

    monkeypatch.setattr(cli, "_write_exclusive", _flaky)
    with pytest.raises(OSError, match="disk full"):
        main(["--table", "dev", "--run-dir", str(run_dir)])
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["exception_type"] == "OSError"
    assert (run_dir / "selection.json").exists()
    assert not (run_dir / "evaluation.json").exists()


def test_write_exclusive_fsyncs_file_and_directory(tmp_path, monkeypatch):
    import os as _os

    calls = []
    real_fsync = _os.fsync
    monkeypatch.setattr(_os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1])
    cli._write_exclusive(tmp_path / "artifact.json", {"a": 1})
    assert len(calls) >= 2  # the file itself, and its containing directory


# --- Astra freeze-review r3 group 1 (2026-09-28): a locked evaluation must
# re-exec into a fresh interpreter (never continue in-process), verifying
# installed dependency versions/hashes before spawning. Mocked so this
# never actually spawns a child or touches locked seeds. ---

def test_cli_reexecs_into_a_fresh_subprocess_for_locked_without_running_it(monkeypatch):
    spawned = []
    dep_checked = []
    real_run = cli.subprocess.run

    def _fake_run(cmd, *args, cwd=None, env=None, **kwargs):
        if cmd[:1] == [sys.executable]:
            spawned.append((cmd, cwd, env))
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, *args, cwd=cwd, env=env, **kwargs)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(cli.lock_guard, "require_dependency_integrity",
                        lambda lock: dep_checked.append(lock))
    monkeypatch.setattr(cli.pipeline, "select",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not run in-process for a locked table")))

    main(["--table", "locked", "--i-have-clearance-to-run-locked-seeds"])

    assert dep_checked  # dependency integrity verified before spawning
    assert len(spawned) == 1
    cmd, cwd, env = spawned[0]
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "runner.cli"]
    assert "--table" in cmd and "locked" in cmd
    assert env[lock_guard.CLEAN_SUBPROCESS_ENV] == "1"


def test_cli_default_run_dir_is_unique_per_invocation():
    """No --run-dir: defaults under the real repo's runs/ (gitignored).
    Cleans up the directories it creates."""
    import shutil

    runs_dir = cli.lock_guard.REPO_ROOT / "runs"
    before = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    try:
        main(["--table", "dev"])
        main(["--table", "dev"])
        after = set(runs_dir.iterdir())
        created = after - before
        assert len(created) == 2
        for d in created:
            assert (d / "evaluation.json").exists()
    finally:
        for d in (set(runs_dir.iterdir()) if runs_dir.is_dir() else set()) - before:
            shutil.rmtree(d, ignore_errors=True)
