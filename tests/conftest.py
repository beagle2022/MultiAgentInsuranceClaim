import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.config import SETTINGS  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return dataclasses.replace(SETTINGS, backend="offline", store_dir=tmp_path / "store", log_dir=tmp_path / "logs")
