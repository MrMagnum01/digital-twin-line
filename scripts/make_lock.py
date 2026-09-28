"""Build experiment-lock.json from the current committed byte content.

Usage: python3 scripts/make_lock.py --protocol <path to frozen-ml-protocol.md>

Reads files only; generates no data, fits nothing, scores nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCKED_SOURCES = ["src/generator.py", "src/ingest.py", "src/features.py", "src/evaluator.py",
                  "config.yaml"]
SUPPORTING_SOURCES = ["src/twin_config.py", "src/schema.sql", "requirements.txt",
                      "requirements.lock", "pytest.ini"]
NOTE = ("Lock only. No evaluation data generated on locked seeds, no model fit, no thresholds "
        "chosen, no results read. Generator/features/evaluator tested only on separate "
        "development seeds and hand-built fixtures.")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_lock(path: Path) -> dict:
    text = path.read_text()
    out = {}
    for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)\s*\\\s*\n\s*--hash=sha256:([0-9a-f]{64})",
                         text, re.M):
        out[m.group(1)] = {"version": m.group(2), "sha256": m.group(3)}
    return out


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", default=str(REPO / "experiment-lock.json"))
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    ref = cfg["plant_ingest_reference"]
    sub_head = git("-C", ref["submodule_path"], "rev-parse", "HEAD")
    if sub_head != ref["commit"]:
        raise SystemExit(f"submodule at {sub_head}, config pins {ref['commit']}")
    deps = parse_lock(REPO / "requirements.lock")
    lock = {
        "lock_version": cfg["version"],
        "protocol_doc": {"path": "vault:10-projects/freelance/launch/twin/frozen-ml-protocol.md",
                         "sha256": sha256(Path(a.protocol))},
        "source_commit": git("rev-parse", "HEAD"),
        "sha256": {p: sha256(REPO / p) for p in LOCKED_SOURCES},
        "supporting_sha256": {p: sha256(REPO / p) for p in SUPPORTING_SOURCES},
        "plant_ingest_reference": {
            "repo": git("config", "-f", ".gitmodules", f"submodule.{ref['submodule_path']}.url"), "commit": ref["commit"], "submodule_path": ref["submodule_path"],
            "reused_symbols": ref["reused_symbols"]},
        "dependency_lock": {
            "file": "requirements.lock",
            "file_sha256": sha256(REPO / "requirements.lock"),
            "install": "pip install --require-hashes --only-binary=:all: -r requirements.lock",
            "platform": "CPython 3.13, manylinux x86_64 wheels",
            "packages": deps,
        },
        "locked_seed_table": cfg["split_table_locked"],
        "development_seeds": sorted(s for sp in cfg["split_table_dev"]["splits"] for s in sp["seeds"]),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "note": NOTE,
    }
    Path(a.out).write_text(json.dumps(lock, indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
