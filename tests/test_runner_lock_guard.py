"""runner.lock_guard: the runner's refusal-to-run gate. Exercised against
the real repo (which must currently verify - HEAD fac94f7, lock-3) and
against hand-built tampered copies in tmp_path; never against a modified
copy of the real locked files in place."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from runner import lock_guard

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _stub_submodule_check(monkeypatch):
    """The tamper fixtures below do not carry a real git submodule; stub the
    submodule-HEAD and submodule-dirty checks to a clean pinned commit so
    these tests isolate the file-hash verification they are actually
    about."""
    monkeypatch.setattr(lock_guard, "_submodule_head",
                        lambda root, path: lock_guard.PINNED_SUBMODULE_COMMIT)
    monkeypatch.setattr(lock_guard, "_submodule_dirty", lambda root, path: False)


def test_verify_lock_passes_on_the_real_repo():
    lock = lock_guard.verify_lock()
    assert lock["lock_version"] == "lock-3"


def test_verify_lock_rejects_a_disagreeing_approved_sha():
    with pytest.raises(lock_guard.LockVerificationError, match="does not match the approved lock"):
        lock_guard.verify_lock(approved_sha256="0" * 64)


def test_verify_lock_rejects_missing_lock_file(tmp_path):
    with pytest.raises(lock_guard.LockVerificationError, match="does not exist"):
        lock_guard.verify_lock(repo_root=tmp_path)


def _copy_repo_subset(tmp_path: Path) -> Path:
    """A tmp_path copy of exactly the files experiment-lock.json hashes,
    unmodified, plus the lock itself. Verifying against this MUST pass
    hash-wise (the sha256 sections match); it is the base every tamper
    test above mutates one file of."""
    lock = json.loads((REPO / "experiment-lock.json").read_bytes())
    (tmp_path / "experiment-lock.json").write_bytes((REPO / "experiment-lock.json").read_bytes())
    for section in ("sha256", "supporting_sha256"):
        for rel in lock[section]:
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / rel, dst)
    return tmp_path


def test_verify_lock_passes_on_an_unmodified_copy(tmp_path):
    _copy_repo_subset(tmp_path)
    lock = lock_guard.verify_lock(repo_root=tmp_path)
    assert lock["lock_version"] == "lock-3"


@pytest.mark.parametrize("rel", ["src/generator.py", "src/features.py", "src/evaluator.py",
                                 "src/ingest.py", "config.yaml", "src/twin_config.py"])
def test_verify_lock_rejects_a_tampered_locked_file(tmp_path, rel):
    _copy_repo_subset(tmp_path)
    target = tmp_path / rel
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    with pytest.raises(lock_guard.LockVerificationError, match="expected sha256"):
        lock_guard.verify_lock(repo_root=tmp_path)


def test_verify_lock_rejects_a_missing_hashed_file(tmp_path):
    _copy_repo_subset(tmp_path)
    (tmp_path / "src/evaluator.py").unlink()
    with pytest.raises(lock_guard.LockVerificationError, match="file missing"):
        lock_guard.verify_lock(repo_root=tmp_path)


def test_verify_lock_rejects_submodule_drift(tmp_path, monkeypatch):
    _copy_repo_subset(tmp_path)
    monkeypatch.setattr(lock_guard, "_submodule_head", lambda root, path: "0" * 40)
    with pytest.raises(lock_guard.LockVerificationError, match="does not match the pinned commit"):
        lock_guard.verify_lock(repo_root=tmp_path)


def test_verify_lock_collects_every_problem_not_just_the_first(tmp_path):
    _copy_repo_subset(tmp_path)
    (tmp_path / "src/generator.py").write_bytes(b"tampered")
    (tmp_path / "src/features.py").write_bytes(b"tampered")
    with pytest.raises(lock_guard.LockVerificationError) as exc:
        lock_guard.verify_lock(repo_root=tmp_path)
    msg = str(exc.value)
    assert "src/generator.py" in msg and "src/features.py" in msg


def test_sha256_helpers_agree_with_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"some bytes")
    assert lock_guard._sha256_file(p) == hashlib.sha256(b"some bytes").hexdigest()


# --- Astra freeze-review MUST-FIX 1 (2026-09-28): verified lock must bind
# what actually executes, not just what verify_lock() hashed on disk. ---

def test_verify_lock_rejects_a_dirty_submodule(tmp_path, monkeypatch):
    _copy_repo_subset(tmp_path)
    monkeypatch.setattr(lock_guard, "_submodule_dirty", lambda root, path: True)
    with pytest.raises(lock_guard.LockVerificationError, match="working tree is dirty"):
        lock_guard.verify_lock(repo_root=tmp_path)


def test_submodule_dirty_detects_real_uncommitted_changes(tmp_path):
    # Bypass this file's autouse _submodule_dirty stub: exercise the real
    # git-backed helper it wraps directly.
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "x"], cwd=tmp_path, check=True)
    assert lock_guard._git_porcelain(tmp_path).strip() == ""
    (tmp_path / "f.txt").write_text("y")
    assert lock_guard._git_porcelain(tmp_path).strip() != ""


def test_require_clean_worktree_passes_clean_and_refuses_dirty(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "x"], cwd=tmp_path, check=True)
    lock_guard.require_clean_worktree(tmp_path)  # no raise: clean
    (tmp_path / "f.txt").write_text("y")
    with pytest.raises(lock_guard.LockVerificationError, match="uncommitted changes"):
        lock_guard.require_clean_worktree(tmp_path)


def test_require_import_paths_rejects_a_repo_root_the_real_module_is_not_under(tmp_path):
    import generator
    with pytest.raises(lock_guard.LockVerificationError, match="imported from"):
        lock_guard.require_import_paths(tmp_path, {"src/generator.py": generator})


def test_require_import_paths_passes_for_the_real_repo():
    import generator
    lock_guard.require_import_paths(lock_guard.REPO_ROOT, {"src/generator.py": generator})


# --- Astra freeze-review r2 group 1 (2026-09-28): the guard must bind the
# config value actually USED, not merely what was hashed on disk earlier -
# no cached or mutated config object may pass through after the guard. ---

def test_load_verified_config_returns_a_deep_frozen_structure(tmp_path):
    _copy_repo_subset(tmp_path)
    lock = lock_guard.verify_lock(repo_root=tmp_path)
    cfg = lock_guard.load_verified_config(tmp_path, lock)
    assert cfg["models"]["isolation_forest"]["params"]["contamination"] == 0.01
    with pytest.raises(TypeError):
        cfg["models"]["isolation_forest"]["params"]["contamination"] = 0.4
    assert isinstance(cfg["models"]["static_threshold"]["k_grid"], tuple)
    with pytest.raises(AttributeError):
        cfg["models"]["static_threshold"]["k_grid"].append(99)  # tuple: no .append


def test_load_verified_config_refuses_when_lock_disagrees_with_config_bytes_at_use(tmp_path):
    _copy_repo_subset(tmp_path)
    lock = json.loads((tmp_path / "experiment-lock.json").read_bytes())
    lock["sha256"]["config.yaml"] = "0" * 64
    with pytest.raises(lock_guard.LockVerificationError, match="config.yaml sha256"):
        lock_guard.load_verified_config(tmp_path, lock)


def test_load_verified_config_is_unaffected_by_a_mutated_twin_config_cache():
    """The literal r2 probe: twin_config.load_config() returns the SAME
    mutable dict on every call in this process. Mutating that cached
    object's contamination to 0.4 must never reach a value load_verified_
    config() hands to a real run."""
    import twin_config

    cached = twin_config.load_config()
    original = cached["models"]["isolation_forest"]["params"]["contamination"]
    cached["models"]["isolation_forest"]["params"]["contamination"] = 0.4
    try:
        lock = lock_guard.require_lock()
        cfg = lock_guard.load_verified_config(lock_guard.REPO_ROOT, lock)
        assert cfg["models"]["isolation_forest"]["params"]["contamination"] == original
    finally:
        cached["models"]["isolation_forest"]["params"]["contamination"] = original


# --- Astra freeze-review r2 group 4 (2026-09-28): a durable, experiment-
# wide reservation must refuse a second locked evaluation regardless of
# --run-dir or caller. Tests the guard directly; never passes allow_locked
# through pipeline/generator and never touches locked seeds. ---

def test_reserve_locked_evaluation_refuses_a_second_reservation_for_the_same_lock(tmp_path):
    _copy_repo_subset(tmp_path)
    marker = lock_guard.reserve_locked_evaluation(tmp_path)
    assert marker.is_file()
    with pytest.raises(lock_guard.LockVerificationError, match="already been reserved"):
        lock_guard.reserve_locked_evaluation(tmp_path)
    # Not scoped to any particular --run-dir: the marker lives outside one,
    # so a caller pointed at a different --run-dir (irrelevant to this
    # call at all) still hits the same, already-claimed reservation.
    with pytest.raises(lock_guard.LockVerificationError, match="already been reserved"):
        lock_guard.reserve_locked_evaluation(tmp_path)
