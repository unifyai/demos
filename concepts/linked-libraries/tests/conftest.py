import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from linked_libraries.generate import build  # noqa: E402


@pytest.fixture(scope="session")
def ds_sanity():
    return build("sanity")


@pytest.fixture(scope="session")
def ds_s():
    return build("S")


@pytest.fixture(scope="session")
def ds_l():
    return build("L")


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
