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

Also exported for callers (runner.pipeline) that must bind what they are
about to EXECUTE, not just what verify_lock() hashed on disk (Astra
freeze-review MUST-FIX 1, 2026-09-28):
  * require_import_paths() - refuses unless every locked module a caller
    has already imported was actually loaded from under the verified
    root, closing the gap where repo_root verifies one directory's bytes
    while Python executes same-named modules imported from elsewhere.
  * require_clean_worktree() - refuses unless `root`'s own git working
    tree has no uncommitted changes. Only meant to gate an actual locked
    run (allow_locked=True): dev/fixture iteration is expected to have a
    dirty tree, and locked-file byte hashes above are already re-checked
    from disk on every call regardless.
  * the submodule check below now also refuses a dirty submodule working
    tree, not just a HEAD mismatch: HEAD alone does not prove nothing in
    the submodule's checkout was locally modified.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import types
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Astra lock-3 sign-off (vault:10-projects/freelance/launch/twin/astra-lock-3-signoff.md):
# "YES -- experiment definition freeze, 2026-09-28. ... experiment-lock.json
#  SHA256: c5f6d8873c1cc91902331c057275b6b9990141ed4b70d7c074ea7f6f2a7f77ad."
APPROVED_LOCK_SHA256 = "c5f6d8873c1cc91902331c057275b6b9990141ed4b70d7c074ea7f6f2a7f77ad"

PINNED_SUBMODULE_COMMIT = "7e378facb410b529d8e2d72d329daf2eee515161"

# Set only by runner.cli's own subprocess re-exec for a locked table, never
# by a caller importing runner.pipeline directly - see
# require_clean_subprocess() (Astra freeze-review r3 group 1, 2026-09-28).
CLEAN_SUBPROCESS_ENV = "TWIN_RUNNER_LOCKED_CLEAN_SUBPROCESS"


class LockVerificationError(RuntimeError):
    """Raised whenever the runner must refuse to proceed: unapproved or
    tampered lock, a locked file that no longer matches its recorded hash,
    or a submodule pin that has drifted."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fsync_dir(dir_path: Path) -> None:
    """fsync the directory entry itself, not just the file inside it: on a
    crash between write() and the directory recording the new dentry, an
    O_EXCL-created file can vanish even though its own fsync succeeded.
    Exclusivity (O_EXCL) proves no concurrent writer raced us; it is not
    crash durability on its own (Astra freeze-review r3 group 4, 2026-09-28)."""
    fd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive_durable(path: Path, payload: bytes) -> None:
    """O_EXCL create-and-write `payload` at `path`, fsync the file, then
    fsync its containing directory. Shared by every durable reservation/
    artifact marker below."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(path.parent)


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


def _git_porcelain(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LockVerificationError(
            f"could not check working-tree cleanliness at {path}: {exc}") from exc
    return out.stdout


def _submodule_dirty(repo_root: Path, submodule_path: str) -> bool:
    return bool(_git_porcelain(repo_root / submodule_path).strip())


def require_clean_worktree(root: Path | str) -> None:
    """Refuse unless `root`'s own git working tree has no uncommitted
    changes (tracked modifications or untracked, non-ignored files).
    Locked-file byte hashes are already re-checked from disk on every
    verify_lock() call; this closes the remaining gap for a genuine
    locked run: an uncommitted change to a file the lock does NOT
    individually hash (e.g. runner/*) could still alter what executes.
    Callers should only invoke this for allow_locked=True runs - dev/
    fixture iteration is expected to have a dirty tree."""
    root = Path(root).resolve()
    dirty = _git_porcelain(root)
    if dirty.strip():
        raise LockVerificationError(
            f"{root} has uncommitted changes, refusing a locked run against a dirty tree: "
            f"{dirty.strip()!r}")


def require_import_paths(root: Path | str, modules: dict) -> None:
    """modules: {relative_path_under_root: already-imported module object}.
    Raises unless every module's own __file__ resolves to exactly that
    path under `root` - i.e. the code this process actually imported and
    will execute is the code whose bytes verify_lock() just hash-checked,
    not a same-named module loaded from a different sys.path entry, a
    stale editable install, or a symlink target outside `root`."""
    root = Path(root).resolve()
    problems: list[str] = []
    for rel, module in modules.items():
        expected = (root / rel).resolve()
        mod_file = getattr(module, "__file__", None)
        if mod_file is None:
            problems.append(f"{module!r} has no __file__: cannot bind its import path")
            continue
        actual = Path(mod_file).resolve()
        if actual != expected:
            problems.append(
                f"{getattr(module, '__name__', module)}: imported from {actual}, expected "
                f"{expected} under the verified repo root")
    if problems:
        raise LockVerificationError(
            "locked-module import-path binding failed, refusing to run: " + "; ".join(problems))


def _deep_freeze(value):
    """dict -> read-only MappingProxyType (values recursively frozen), list
    -> tuple (elements recursively frozen), everything else unchanged. Used
    by load_verified_config() so nothing downstream can mutate the config
    object a run actually executes against."""
    if isinstance(value, dict):
        return types.MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def load_verified_config(root: Path | str, lock: dict) -> types.MappingProxyType:
    """Return config.yaml as a deep-frozen structure, loaded fresh from
    disk bytes and re-verified against `lock`'s own recorded config.yaml
    sha256 at this point of use - never through twin_config.load_config(),
    whose @lru_cache returns the SAME mutable dict object on every call in
    this process: a caller that obtains that object (e.g. via
    twin_config.load_config() elsewhere) and mutates it - `cfg[...]
    ["contamination"] = 0.4` - permanently corrupts every future call
    inside this process, including one made immediately after require_lock()
    verified the on-disk bytes were untouched (Astra freeze-review r2 group
    1, 2026-09-28). Loading fresh bytes, re-checking their hash right here,
    and freezing the result closes that gap regardless of what any other
    holder of a twin_config-cached dict has done to their copy."""
    root = Path(root).resolve()
    path = root / "config.yaml"
    if not path.is_file():
        raise LockVerificationError(f"{path} does not exist: cannot load the verified config")
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    expected = lock.get("sha256", {}).get("config.yaml")
    if expected is None or actual != expected:
        raise LockVerificationError(
            f"config.yaml sha256 {actual} does not match the locked value {expected!r} at the "
            "point of use: refusing to run against an unverified or tampered config")
    import yaml

    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        raise LockVerificationError(f"{path} is not a mapping")
    return _deep_freeze(cfg)


def reserve_locked_evaluation(root: Path | str) -> Path:
    """Durable, experiment-wide reservation for a locked evaluation, keyed
    by experiment-lock.json's own sha256 - deliberately NOT under any
    caller-chosen --run-dir, so a second locked evaluation cannot obtain a
    fresh reservation merely by pointing at a different --run-dir, nor by
    calling runner.pipeline directly instead of the CLI (Astra freeze-
    review r2 group 4, 2026-09-28: "a different --run-dir or direct
    pipeline call can repeat it"). O_EXCL: the first locked evaluation for
    this experiment-lock identity claims the marker; every later attempt -
    from any run-dir, any caller, forever - refuses."""
    root = Path(root).resolve()
    lock_sha = _sha256_file(root / "experiment-lock.json")
    marker_dir = root / "runs" / "_locked_reservations"
    marker = marker_dir / f"{lock_sha}.reserved"
    payload = json.dumps({
        "reserved_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "experiment_lock_sha256": lock_sha,
    }, indent=1, sort_keys=True).encode("utf-8")
    try:
        _write_exclusive_durable(marker, payload)
    except FileExistsError as exc:
        raise LockVerificationError(
            f"a locked evaluation for experiment-lock.json sha256 {lock_sha} has already been "
            f"reserved at {marker}: refusing a second locked evaluation for the same "
            "experiment-lock identity") from exc
    return marker


def require_clean_subprocess() -> None:
    """Refuse unless this process was started specifically to run a locked
    evaluation by runner.cli's own subprocess re-exec, which sets
    CLEAN_SUBPROCESS_ENV right before spawning a fresh interpreter (never
    set by a caller that merely imports runner.pipeline). Import-path
    equality (require_import_paths) only proves where an already-imported
    module's __file__ resolves; it says nothing about whether that already-
    loaded bytecode was altered after import in a long-lived process. A
    freshly started interpreter that then imports the locked modules for
    the first time closes that gap (Astra freeze-review r3 group 1,
    2026-09-28: "import-path equality alone is not verification of already-
    loaded code bytes")."""
    if os.environ.get(CLEAN_SUBPROCESS_ENV) != "1":
        raise LockVerificationError(
            "a locked evaluation must run via `python -m runner.cli --table locked ...`, which "
            "re-execs into a fresh interpreter and verifies installed dependency versions/hashes "
            "before importing any locked module: refusing an in-process pipeline.select"
            "(allow_locked=True) call made outside that fresh subprocess")


def require_dependency_integrity(lock: dict) -> None:
    """Refuse unless every package experiment-lock.json's dependency_lock
    pins is actually installed, at exactly the locked version, in THIS
    interpreter's environment, and every file RECORD (pip's own install
    manifest) lists for it still hashes to what pip recorded at install
    time. This is deliberately not a re-derivation of dependency_lock's
    own wheel sha256 (there is no reliable, offline way to recompute the
    original PyPI wheel hash from files already unpacked on disk); it is
    the honest, available substitute - installed version plus on-disk
    file integrity against the package's own install record - so a locked
    run refuses against a drifted or locally-tampered dependency instead
    of only recording package versions after the fact (Astra freeze-review
    r3 group 1, 2026-09-28: "installed dependency versions are recorded but
    not checked")."""
    import importlib.metadata as importlib_metadata

    packages = lock.get("dependency_lock", {}).get("packages", {})
    problems: list[str] = []
    for name, spec in packages.items():
        expected_version = spec.get("version")
        try:
            dist = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            problems.append(f"{name}: not installed in this interpreter's environment")
            continue
        if dist.version != expected_version:
            problems.append(
                f"{name}: installed version {dist.version} does not match locked version "
                f"{expected_version}")
            continue
        record = dist.read_text("RECORD")
        if record is None:
            problems.append(f"{name}: no RECORD available to verify installed file integrity")
            continue
        base = Path(str(dist.locate_file("")))
        for line in record.splitlines():
            parts = line.split(",")
            if len(parts) < 3 or not parts[1]:
                continue
            rel_path, hash_field = parts[0], parts[1]
            if not hash_field.startswith("sha256="):
                continue
            file_path = base / rel_path
            if not file_path.is_file():
                problems.append(f"{name}: RECORD-listed file missing on disk: {rel_path}")
                continue
            digest = hashlib.sha256(file_path.read_bytes()).digest()
            actual = "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            if actual != hash_field:
                problems.append(
                    f"{name}: installed file {rel_path} no longer matches its own install "
                    "RECORD hash")
    if problems:
        raise LockVerificationError(
            "installed dependency verification failed, refusing a locked run: " + "; ".join(problems))


def _selection_marker_payload(selection_id: str, extra: dict) -> bytes:
    payload = {"selection_id": selection_id, "recorded_at": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), **extra}
    return json.dumps(payload, indent=1, sort_keys=True, default=str).encode("utf-8")


def persist_selection_durably(root: Path | str, selection_id: str, selection: dict) -> Path:
    """Durably record a validation-only selection BEFORE any test-split
    scoring may occur for it, keyed by the selection's own content hash
    (`selection_id`) - independent of any caller-chosen --run-dir, so a
    direct runner.pipeline call (not just runner.cli) also leaves this
    record. score_test() refuses to score a locked selection that has no
    matching record here (Astra freeze-review r3 group 4, 2026-09-28:
    "pipeline.run called directly must never score test before selection is
    durably persisted")."""
    root = Path(root).resolve()
    marker = root / "runs" / "_selections" / f"{selection_id}.json"
    payload = _selection_marker_payload(selection_id, {"selection": selection})
    try:
        _write_exclusive_durable(marker, payload)
    except FileExistsError as exc:
        raise LockVerificationError(
            f"a selection for identity {selection_id} is already durably persisted at {marker}: "
            "refusing to persist a second, possibly different, record for the same identity") from exc
    return marker


def require_persisted_selection(root: Path | str, selection_id: str) -> None:
    """Refuse unless persist_selection_durably() already wrote a durable
    record for exactly this selection_id - the immutable, content-hashed
    identity of the selection score_test() is about to score."""
    root = Path(root).resolve()
    marker = root / "runs" / "_selections" / f"{selection_id}.json"
    if not marker.is_file():
        raise LockVerificationError(
            f"no durably persisted selection record found for identity {selection_id} at "
            f"{marker}: refusing to score test before the validation-only selection is on "
            "durable storage")


def reserve_selection_scoring(root: Path | str, selection_id: str) -> Path:
    """One-shot, durable claim that this exact persisted selection identity
    is being scored on test - refuses a second scoring of the SAME
    selection, from any caller, forever (Astra freeze-review r3 group 4,
    2026-09-28: "score_test must accept only the one persisted, immutable
    selection ... refuse repeats or mutation")."""
    root = Path(root).resolve()
    marker = root / "runs" / "_scored_selections" / f"{selection_id}.scored"
    payload = _selection_marker_payload(selection_id, {})
    try:
        _write_exclusive_durable(marker, payload)
    except FileExistsError as exc:
        raise LockVerificationError(
            f"selection identity {selection_id} has already been scored once at {marker}: "
            "refusing to repeat test scoring for the same persisted, immutable selection") from exc
    return marker


def verify_lock(repo_root: Path | str | None = None,
                approved_sha256: str | None = None) -> dict:
    """Load and verify experiment-lock.json plus every file it hashes.
    Returns the parsed lock dict on success. Raises LockVerificationError,
    collecting every problem found, rather than stopping at the first one."""
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
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
            elif _submodule_dirty(root, submodule_path):
                problems.append(
                    f"submodule {submodule_path} working tree is dirty despite HEAD matching "
                    "the pinned commit: HEAD alone does not prove its checked-out files were "
                    "not locally modified")

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
