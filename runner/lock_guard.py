"""Lock verification gate (NOT a locked file itself; guards them).

Every runner entry point calls require_lock() before touching any locked
input. It refuses (raises LockVerificationError) unless:

  1. experiment-lock.json's own sha256 equals the sha Astra signed off in
     vault:10-projects/freelance/launch/twin/astra-lock-3-signoff.md
     ("YES ... experiment-lock.json SHA256: c5f6d887...").
  2. Every file hash experiment-lock.json records (its "sha256" and
     "supporting_sha256" sections: generator/ingest/features/evaluator/
     config, plus twin_config.py, schema.sql, requirements.txt/.lock,
     pytest.ini) still matches the bytes on disk.
  3. The pinned plant-ingest submodule's actual git HEAD matches both
     config.yaml's and experiment-lock.json's recorded commit.

There is no partial or best-effort pass: any mismatch, any missing file, or
git being unavailable to check the submodule pin, is a hard refusal.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Astra lock-3 sign-off (vault:10-projects/freelance/launch/twin/astra-lock-3-signoff.md):
# "YES -- experiment definition freeze, 2026-09-28. ... experiment-lock.json
#  SHA256: c5f6d8873c1cc91902331c057275b6b9990141ed4b70d7c074ea7f6f2a7f77ad."
APPROVED_LOCK_SHA256 = "c5f6d8873c1cc91902331c057275b6b9990141ed4b70d7c074ea7f6f2a7f77ad"

PINNED_SUBMODULE_COMMIT = "7e378facb410b529d8e2d72d329daf2eee515161"


class LockVerificationError(RuntimeError):
    """Raised whenever the runner must refuse to proceed: unapproved or
    tampered lock, a locked file that no longer matches its recorded hash,
    or a submodule pin that has drifted."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _submodule_head(repo_root: Path, submodule_path: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root / submodule_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LockVerificationError(
            f"could not verify the pinned submodule at {submodule_path}: {exc}") from exc
    return out.stdout.strip()


def verify_lock(repo_root: Path | str | None = None,
                approved_sha256: str | None = None) -> dict:
    """Load and verify experiment-lock.json plus every file it hashes.
    Returns the parsed lock dict on success. Raises LockVerificationError,
    collecting every problem found, rather than stopping at the first one."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    approved = approved_sha256 or APPROVED_LOCK_SHA256
    lock_path = root / "experiment-lock.json"
    if not lock_path.is_file():
        raise LockVerificationError(
            f"{lock_path} does not exist: refusing to run without an approved experiment lock")

    raw = lock_path.read_bytes()
    actual = _sha256_bytes(raw)
    if actual != approved:
        raise LockVerificationError(
            f"experiment-lock.json sha256 {actual} does not match the approved lock {approved}: "
            "refusing to run against an unapproved or modified experiment definition")

    lock = json.loads(raw)
    problems: list[str] = []
    for section in ("sha256", "supporting_sha256"):
        for rel, expected in lock.get(section, {}).items():
            path = root / rel
            if not path.is_file():
                problems.append(f"{rel}: file missing")
                continue
            got = _sha256_file(path)
            if got != expected:
                problems.append(f"{rel}: expected sha256 {expected}, got {got}")

    ref = lock.get("plant_ingest_reference", {})
    submodule_path = ref.get("submodule_path")
    recorded_commit = ref.get("commit")
    if submodule_path and recorded_commit:
        if recorded_commit != PINNED_SUBMODULE_COMMIT:
            problems.append(
                f"experiment-lock.json plant_ingest_reference.commit {recorded_commit} "
                f"does not match the pin this runner was built against {PINNED_SUBMODULE_COMMIT}")
        else:
            head = _submodule_head(root, submodule_path)
            if head != recorded_commit:
                problems.append(
                    f"submodule {submodule_path} HEAD {head} does not match the pinned "
                    f"commit {recorded_commit} recorded in experiment-lock.json")

    if problems:
        raise LockVerificationError(
            "locked-file integrity check failed, refusing to run: " + "; ".join(problems))
    return lock


def require_lock(repo_root: Path | str | None = None,
                 approved_sha256: str | None = None) -> dict:
    """Call at the top of any runner entry point that touches locked
    inputs. Same as verify_lock(); kept as a separate name so call sites
    read as a precondition, not a lookup."""
    return verify_lock(repo_root, approved_sha256)
