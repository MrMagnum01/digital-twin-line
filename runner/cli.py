"""CLI entry point for runner.pipeline.run().

Refuses the locked seed table unless BOTH the caller passes
`--table locked` AND the explicit `--i-have-clearance-to-run-locked-seeds`
flag - and even then, lock_guard.require_lock() and
generator.generate()'s own LockedSeedsError refusal still apply
underneath. Nothing in this milestone's tests, dashboard or monitoring
code passes that flag; it exists only so a future, separately authorised
locked run does not require editing this file.

Every invocation writes exclusive (O_EXCL), never-overwritten artifacts
under --run-dir, so a report can never be silently overwritten, rerun
under the same identity, or lost between printing to stdout and actually
landing on disk (Astra freeze-review MUST-FIX 4, 2026-09-28):
  pre_run.json    - runner-implementation hashes, experiment-lock sha256/
                    version, the seed table this run is about to touch,
                    host and dependency versions, written before any dev
                    or locked generation happens (Astra freeze-review r4
                    group 4, 2026-09-28: lock/seeds/runtime binding
                    completed, not left as the earlier short dict).
  selection.json  - the validation-only candidate tables and chosen
                    thresholds/selections for both models, written via
                    pipeline.select() strictly BEFORE pipeline.score_test()
                    makes any test-split score_stratum call (Astra freeze-
                    review r2 group 4, 2026-09-28 - selection.json used to
                    be written only after test was already scored).
  evaluation.json - the full report (pipeline.select() + score_test()).
  failure.json    - written instead of evaluation.json if either phase
                    raises, recording the exception so a failed run is
                    never just silently absent.
Re-running against the same --run-dir refuses at the first artifact write
(pre_run.json) rather than overwriting or silently reusing that identity.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from runner import lock_guard, pipeline

_RUNNER_IMPLEMENTATION_FILES = (
    "runner/pipeline.py", "runner/lock_guard.py", "runner/models.py", "runner/cli.py",
)


class ArtifactExistsError(RuntimeError):
    """Raised when a run artifact already exists: refusing to overwrite or
    silently reuse a previously evaluated run identity."""


def _write_exclusive(path: Path, data: dict) -> None:
    lock_guard._mkdir_durable(path.parent)
    text = json.dumps(data, indent=1, sort_keys=True, default=str)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise ArtifactExistsError(
            f"{path} already exists: refusing to overwrite a run artifact") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    lock_guard._fsync_dir(path.parent)


def _record_failure(run_dir: Path, exc: BaseException) -> None:
    failure = {"failed_at": datetime.now(timezone.utc).isoformat(),
              "exception_type": type(exc).__name__, "exception_message": str(exc)}
    _write_exclusive(run_dir / "failure.json", failure)


def _dependency_versions() -> dict:
    import numpy
    import scipy
    import sklearn
    return {"python": sys.version, "numpy": numpy.__version__,
           "scikit_learn": sklearn.__version__, "scipy": scipy.__version__}


def _git_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _isolated_bootstrap(root: Path) -> str:
    """A minimal -c bootstrap for the fresh, isolated (-I) child
    interpreter: explicitly puts only the verified repo root on sys.path
    before importing runner.cli - never relying on `-m`'s ambient cwd
    insertion (which -I removes, so plain `-I -m runner.cli` cannot even
    find the package) or on any inherited PYTHONPATH. This is the "with
    controlled imports" half of the isolation fix: the child's own sys.path
    is built by us, from the one root require_lock() just verified, not by
    ambient interpreter conventions (Astra freeze-review r4 group 1,
    2026-09-28)."""
    return f"import sys; sys.path.insert(0, {str(root)!r}); from runner.cli import main; main()"


def _reexec_locked_in_fresh_subprocess(argv: list) -> None:
    """A locked evaluation must execute from a freshly started interpreter,
    never inside this (or any other) already-running process whose already-
    imported runner/src modules could have been altered after import -
    import-path equality only proves where a module's __file__ resolves,
    never that its already-loaded bytecode still matches those bytes (Astra
    freeze-review r3 group 1, 2026-09-28). Verifies installed dependency
    versions/hashes against experiment-lock.json's dependency_lock BEFORE
    spawning, so a drifted environment is refused without even starting the
    child, then re-execs into a fresh interpreter running in isolated mode
    (-I: ignores PYTHONPATH/PYTHONHOME and user site-packages, and disables
    sitecustomize/usercustomize processing) with an explicit, controlled
    sys.path built from the verified root (see _isolated_bootstrap) -
    closing the gap where `-m runner.cli` alone still silently inherited an
    arbitrary caller PYTHONPATH able to shadow imports or run arbitrary
    startup code (Astra freeze-review r4 group 1, 2026-09-28: "the child
    interpreter must run in isolated mode ... no inherited PYTHONPATH or
    user site"). The env passed to the child additionally strips every
    PYTHON* variable as a second, independent layer - belt and braces, not
    a substitute for -I. Marked via lock_guard.CLEAN_SUBPROCESS_ENV so
    pipeline.select() knows this is that fresh child, not a direct caller."""
    root = lock_guard.REPO_ROOT
    lock = lock_guard.require_lock(root)
    lock_guard.require_dependency_integrity(lock)
    env = lock_guard.isolated_subprocess_env(os.environ)
    env[lock_guard.CLEAN_SUBPROCESS_ENV] = "1"
    cmd = [sys.executable, "-I", "-c", _isolated_bootstrap(root), *argv]
    result = subprocess.run(cmd, cwd=str(root), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main(argv=None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", choices=["dev", "locked"], default="dev")
    p.add_argument("--i-have-clearance-to-run-locked-seeds", action="store_true",
                   dest="allow_locked",
                   help="Required in addition to --table locked. Do not pass this without "
                        "a recorded freeze-clearance decision (see "
                        "vault:10-projects/freelance/launch/twin/astra-lock-3-signoff.md).")
    p.add_argument("--run-dir", default=None,
                   help="directory for this run's exclusive artifacts (pre_run.json, "
                        "selection.json, evaluation.json / failure.json); must not already "
                        "contain any of them. Default: runs/<table>-<UTC timestamp>-<pid> "
                        "under the repo root.")
    args = p.parse_args(argv)

    if args.table == "locked" and not args.allow_locked:
        p.error("--table locked requires --i-have-clearance-to-run-locked-seeds")

    if args.table == "locked" and os.environ.get(lock_guard.CLEAN_SUBPROCESS_ENV) != "1":
        _reexec_locked_in_fresh_subprocess(raw_argv)
        return

    started = datetime.now(timezone.utc)
    root = lock_guard.REPO_ROOT
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / "runs" / f"{args.table}-{stamp}-{os.getpid()}"

    try:
        # Lock/seeds/runtime binding is established HERE, before any
        # generation happens, not left as "pre_run records package versions
        # rather than enforcing them" (Astra freeze-review r3 group 4). A
        # bad lock now fails inside this same try/except, so the attempt
        # still leaves a failure.json record rather than aborting with no
        # trace at all - re-verified again inside pipeline.select() itself
        # regardless, per that module's own point-of-use philosophy.
        lock = lock_guard.require_lock(root)
        cfg = lock_guard.load_verified_config(root, lock)
        seeds = sorted({s for sp in cfg[f"split_table_{args.table}"]["splits"] for s in sp["seeds"]})
        pre_run = {
            "started_at": started.isoformat(),
            "table": args.table,
            "allow_locked": args.allow_locked,
            "repo_root": str(root),
            "repo_head": _git_head(root),
            "experiment_lock_sha256": lock_guard._sha256_file(root / "experiment-lock.json"),
            "lock_version": lock.get("lock_version"),
            "seeds": seeds,
            "runner_implementation_sha256": {
                rel: lock_guard._sha256_file(root / rel) for rel in _RUNNER_IMPLEMENTATION_FILES
            },
            "host": platform.node(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
        }
        _write_exclusive(run_dir / "pre_run.json", pre_run)
    except ArtifactExistsError:
        raise
    except BaseException as exc:
        try:
            _record_failure(run_dir, exc)
        except BaseException:
            pass
        raise

    try:
        sel = pipeline.select(args.table, allow_locked=args.allow_locked)
    except BaseException as exc:
        _record_failure(run_dir, exc)
        raise

    selection = {
        "table": sel.table,
        "lock_version": sel.lock.get("lock_version"),
        "dataset_hash": sel.dataset_hash,
        "static_threshold_selection": sel.static_threshold_report["selection"],
        "static_threshold_chosen_k": sel.static_threshold_report["chosen_k"],
        "isolation_forest_selection": sel.isolation_forest_report["selection"],
        "isolation_forest_chosen_threshold": sel.isolation_forest_report["chosen_threshold"],
    }
    try:
        # Written before score_test() below makes any test-split
        # score_stratum call: the validation-only selection is on durable
        # storage first. A write failure here must itself surface as a
        # failure record, not silently leave the run's only trace as
        # whatever score_test() does next (r3 group 4).
        _write_exclusive(run_dir / "selection.json", selection)
    except BaseException as exc:
        try:
            _record_failure(run_dir, exc)
        except BaseException:
            pass
        raise

    try:
        report = pipeline.score_test(sel)
    except BaseException as exc:
        _record_failure(run_dir, exc)
        raise

    try:
        _write_exclusive(run_dir / "evaluation.json", report)
    except BaseException as exc:
        try:
            _record_failure(run_dir, exc)
        except BaseException:
            pass
        raise
    print(json.dumps(report, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
