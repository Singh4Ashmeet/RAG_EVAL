import sys


def test_python_version():
    assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version_info}"


def test_import_core():
    import core  # noqa: F401
