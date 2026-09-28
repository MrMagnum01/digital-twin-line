"""Lock integrity: once experiment-lock.json exists, every hashed file must
still match it, and the pinned references must agree."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "experiment-lock.json"


def test_config_pins_the_submodule_commit():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    ref = cfg["plant_ingest_reference"]
    head = subprocess.run(["git", "-C", str(REPO / ref["submodule_path"]), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == ref["commit"] == "7e378facb410b529d8e2d72d329daf2eee515161"


@pytest.mark.skipif(not LOCK.exists(), reason="experiment-lock.json not generated yet")
def test_lock_hashes_match_files():
    lock = json.loads(LOCK.read_text())
    for rel, h in {**lock["sha256"], **lock["supporting_sha256"]}.items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == h, rel
    assert set(lock["sha256"]) >= {"src/generator.py", "src/ingest.py", "src/features.py",
                                   "src/evaluator.py", "config.yaml"}
    assert lock["dependency_lock"]["file_sha256"] == \
        hashlib.sha256((REPO / "requirements.lock").read_bytes()).hexdigest()
    assert lock["plant_ingest_reference"]["commit"] == "7e378facb410b529d8e2d72d329daf2eee515161"
    assert "No evaluation data generated on locked seeds" in lock["note"]
