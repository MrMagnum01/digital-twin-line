"""Test setup. All generated data here comes from the DEVELOPMENT split
table (seeds 901-906, dates from 2023-09-04) - never the locked table."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

DEV_SEEDS = {901, 902, 903, 904, 905, 906}


@pytest.fixture(scope="session")
def cfg():
    from twin_config import load_config
    return load_config()


@pytest.fixture(scope="session")
def dev_ds():
    import generator
    ds = generator.generate("dev")
    seeds = {s for sp in ds.splits.values() for s in sp.seeds}
    assert seeds == DEV_SEEDS, "tests must only ever use development seeds"
    return ds
