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
    submodule-HEAD check to the pinned commit so these tests isolate the
    file-hash verification they are actually about."""
    monkeypatch.setattr(lock_guard, "_submodule_head",
                        lambda root, path: lock_guard.PINNED_SUBMODULE_COMMIT)


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
