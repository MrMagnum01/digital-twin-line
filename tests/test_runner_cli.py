"""runner.cli: argument-level refusal only (dev-table happy path is already
covered end to end by test_runner_pipeline.py)."""
import json

import pytest

from runner.cli import main


def test_cli_refuses_locked_table_without_the_clearance_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--table", "locked"])
    assert exc.value.code == 2
    assert "requires --i-have-clearance-to-run-locked-seeds" in capsys.readouterr().err


def test_cli_writes_a_json_report_for_the_dev_table(tmp_path):
    out = tmp_path / "report.json"
    main(["--table", "dev", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["table"] == "dev"
    assert report["lock_version"] == "lock-3"
