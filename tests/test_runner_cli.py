"""runner.cli: argument-level refusal and exclusive-artifact behaviour only
(dev-table happy path itself is already covered end to end by
test_runner_pipeline.py)."""
import json
import os
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
    assert cmd[1] == "-I"
    assert cmd[2] == "-c"
    assert "runner.cli" in cmd[3]
    assert "--table" in cmd and "locked" in cmd
    assert env[lock_guard.CLEAN_SUBPROCESS_ENV] == "1"


# --- Astra freeze-review r4 group 1 (2026-09-28): "runner/cli.py:107-109
# copies the caller environment and launches without isolation. The probe
# confirms arbitrary PYTHONPATH is inherited." The fix must run the child
# with -I AND strip PYTHON* vars from its env - checked here at the mocked
# subprocess.run() boundary; test_isolated_mode_actually_ignores_an_
# injected_pythonpath below verifies -I's effect with a real interpreter. ---

def test_locked_reexec_runs_isolated_and_strips_python_env_vars(monkeypatch):
    spawned = []
    real_run = cli.subprocess.run

    def _fake_run(cmd, *args, cwd=None, env=None, **kwargs):
        if cmd[:1] == [sys.executable]:
            spawned.append((cmd, cwd, env))
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, *args, cwd=cwd, env=env, **kwargs)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(cli.lock_guard, "require_dependency_integrity", lambda lock: None)
    monkeypatch.setenv("PYTHONPATH", "/tmp/unreviewed-imports")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/unreviewed-startup")

    cli._reexec_locked_in_fresh_subprocess([])

    assert len(spawned) == 1
    cmd, cwd, env = spawned[0]
    assert "-I" in cmd
    assert "PYTHONPATH" not in env
    assert "PYTHONSTARTUP" not in env


def test_isolated_mode_actually_ignores_an_injected_pythonpath(tmp_path):
    """Not mocked: a real child interpreter, launched with -I exactly as
    _reexec_locked_in_fresh_subprocess launches one, must not execute code
    from an attacker/caller-controlled PYTHONPATH entry (e.g. a
    sitecustomize.py) - closing the gap the r4 probe demonstrated
    (env['PYTHONPATH'] reaching the child unchanged) at the interpreter
    level, not just in the argv/env this process happens to construct."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    marker = tmp_path / "sitecustomize_ran"
    (shadow / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shadow)
    subprocess.run([sys.executable, "-I", "-c", "pass"], cwd=str(shadow), env=env, check=True)
    assert not marker.exists()


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


# --- Astra freeze-review r4 group 4 (2026-09-28): "The previously required
# pre-run lock/seeds/runtime binding ... must still be completed ...
# pre_run remains the earlier short metadata dictionary." ---

def test_pre_run_binds_lock_hash_version_and_seeds(tmp_path):
    run_dir = tmp_path / "run_prerun"
    main(["--table", "dev", "--run-dir", str(run_dir)])
    pre_run = json.loads((run_dir / "pre_run.json").read_text())
    assert pre_run["lock_version"] == "lock-3"
    assert pre_run["experiment_lock_sha256"] == lock_guard.APPROVED_LOCK_SHA256
    assert set(pre_run["seeds"]) == {901, 902, 903, 904, 905, 906}


def test_cli_records_a_failure_artifact_when_the_lock_itself_fails_verification(tmp_path, monkeypatch):
    """Binding the lock hash/version/seeds into pre_run now runs require_lock()
    before pre_run.json is written; a bad lock must still leave a failure
    record, never abort with no artifact trace at all."""
    run_dir = tmp_path / "run_badlock"
    monkeypatch.setattr(cli.lock_guard, "APPROVED_LOCK_SHA256", "0" * 64)
    with pytest.raises(lock_guard.LockVerificationError):
        main(["--table", "dev", "--run-dir", str(run_dir)])
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["exception_type"] == "LockVerificationError"
    assert not (run_dir / "pre_run.json").exists()
