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
  pre_run.json    - runner-implementation hashes, table/lock identity,
                    host and dependency versions, written before any dev
                    or locked generation happens.
  selection.json  - the validation-only candidate tables and chosen
                    thresholds/selections for both models.
  evaluation.json - the full pipeline.run() report.
  failure.json    - written instead of evaluation.json if the run raises,
                    recording the exception so a failed run is never just
                    silently absent.
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

from runner import lock_guard
from runner.pipeline import run

_RUNNER_IMPLEMENTATION_FILES = (
    "runner/pipeline.py", "runner/lock_guard.py", "runner/models.py", "runner/cli.py",
)


class ArtifactExistsError(RuntimeError):
    """Raised when a run artifact already exists: refusing to overwrite or
    silently reuse a previously evaluated run identity."""


def _write_exclusive(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, sort_keys=True, default=str)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise ArtifactExistsError(
            f"{path} already exists: refusing to overwrite a run artifact") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


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


def main(argv=None) -> None:
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

    started = datetime.now(timezone.utc)
    root = lock_guard.REPO_ROOT
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / "runs" / f"{args.table}-{stamp}-{os.getpid()}"

    pre_run = {
        "started_at": started.isoformat(),
        "table": args.table,
        "allow_locked": args.allow_locked,
        "repo_root": str(root),
        "repo_head": _git_head(root),
        "runner_implementation_sha256": {
            rel: lock_guard._sha256_file(root / rel) for rel in _RUNNER_IMPLEMENTATION_FILES
        },
        "host": platform.node(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
    }
    _write_exclusive(run_dir / "pre_run.json", pre_run)

    try:
        report = run(args.table, allow_locked=args.allow_locked)
    except BaseException as exc:
        failure = {"failed_at": datetime.now(timezone.utc).isoformat(),
                  "exception_type": type(exc).__name__, "exception_message": str(exc)}
        _write_exclusive(run_dir / "failure.json", failure)
        raise

    selection = {
        "table": report["table"],
        "lock_version": report["lock_version"],
        "dataset_hash": report["dataset_hash"],
        "static_threshold_selection": report["static_threshold"]["selection"],
        "static_threshold_chosen_k": report["static_threshold"]["chosen_k"],
        "isolation_forest_selection": report["isolation_forest"]["selection"],
        "isolation_forest_chosen_threshold": report["isolation_forest"]["chosen_threshold"],
    }
    _write_exclusive(run_dir / "selection.json", selection)
    _write_exclusive(run_dir / "evaluation.json", report)
    print(json.dumps(report, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
