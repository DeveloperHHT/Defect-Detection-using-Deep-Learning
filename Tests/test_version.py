import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))
from bump_version import bump, next_version  # noqa: E402


def test_patch_counts_up():
    assert next_version("0.0.1") == "0.0.2"


def test_patch_carries_into_minor():
    assert next_version("0.0.9") == "0.1.0"


def test_minor_carries_into_major():
    assert next_version("0.9.9") == "1.0.0"


def test_bump_rewrites_file(tmp_path):
    file = tmp_path / "version.py"
    file.write_text('__version__ = "0.0.9"\n')
    assert bump(file) == "0.1.0"
    assert '"0.1.0"' in file.read_text()
