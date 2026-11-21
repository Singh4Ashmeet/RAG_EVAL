import sys


def test_python_version():
    assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version_info}"


def test_import_core():
    import core  # noqa: F401


def test_production_rag_smoke_test():
    import main

    assert main.run_production_smoke_test() is True
