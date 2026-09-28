"""dashboard.app: a real (short-window, dev-seed-only) demo DB, driven
headlessly through Streamlit's own AppTest harness - no browser available
in this environment, but AppTest actually executes app.py's script and
widget tree, which is Streamlit's documented way to test an app without one."""
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    from dashboard.seed_demo_db import seed

    db_path = tmp_path_factory.mktemp("dash") / "twin_demo.duckdb"
    seed(db_path, window_hours=0.75, model_warmup_windows=30)
    return db_path


def test_app_runs_without_exceptions_against_a_seeded_db(demo_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TWIN_DB_PATH", str(demo_db))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py"))
    at.run(timeout=60)
    assert not at.exception

    titles = [t.value for t in at.title]
    assert any("Digital twin line" in t for t in titles)
    captions = [c.value for c in at.caption]
    assert any("independent review pending" in c for c in captions)
    assert any("No client data or client work" in c for c in captions)


def test_app_shows_an_error_for_a_missing_db_file(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TWIN_DB_PATH", str(tmp_path / "does_not_exist.duckdb"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py"))
    at.run(timeout=60)
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("seed_demo_db.py" in e for e in errors)
