"""Tests for the simulated-cluster check in 03_export.py.

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
THREE = [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def _clusters(sim_flags, scene_modes=THREE):
    return [
        {"id": f"c{i}", "simulated": f, **({"scene_modes": scene_modes} if f else {})}
        for i, f in enumerate(sim_flags)
    ]


def test_one_simulated_cluster_with_three_modes_passes():
    assert export.check_simulated_count(_clusters([True, False, False])) == 1


def test_no_simulated_cluster_fails():
    with pytest.raises(SystemExit) as e:
        export.check_simulated_count(_clusters([False, False]))
    assert "0 cluster" in str(e.value)


def test_two_simulated_clusters_fail():
    with pytest.raises(SystemExit) as e:
        export.check_simulated_count(_clusters([True, True, False]))
    assert "2 cluster" in str(e.value)


def test_wrong_number_of_scene_modes_fails():
    with pytest.raises(SystemExit) as e:
        export.check_simulated_count(_clusters([True, False], scene_modes=THREE[:2]))
    assert "2 scene mode" in str(e.value)
