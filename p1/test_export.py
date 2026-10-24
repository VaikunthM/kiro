"""Tests for the names.json simulated-cluster count check in 03_export.py.

Run from p1/:  python3 -m pytest test_export.py -v
"""
import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_export():
    spec = importlib.util.spec_from_file_location("export_mod", os.path.join(_HERE, "03_export.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export = _load_export()


def _names(flags):
    return {str(i): {"id": f"cluster_{i}", "name": f"c{i}", "simulate": f} for i, f in enumerate(flags)}


def test_exactly_three_passes():
    assert export.check_simulated_count(_names([True, True, True, False, False])) == 3


def test_too_few_fails():
    with pytest.raises(SystemExit) as e:
        export.check_simulated_count(_names([True, True, False, False]))
    assert "2 cluster" in str(e.value) and "names.json" in str(e.value)


def test_too_many_fails():
    with pytest.raises(SystemExit) as e:
        export.check_simulated_count(_names([True, True, True, True]))
    assert "4 cluster" in str(e.value)


def test_missing_simulate_key_counts_as_false():
    names = {str(i): {"id": f"cluster_{i}"} for i in range(5)}
    with pytest.raises(SystemExit) as e:
        export.check_simulated_count(names)
    assert "0 cluster" in str(e.value)
