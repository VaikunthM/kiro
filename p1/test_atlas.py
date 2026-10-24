"""Property-based tests (Hypothesis) for the Failure Atlas.

Run from p1/:  python3 -m pytest test_atlas.py -q
"""
import json
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modes import NAMES, SIMULATED, mode_of

_HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = json.load(open(os.path.join(_HERE, "site", "atlas.json")))
IDS = {c["id"] for c in ATLAS["clusters"]}

anything = st.one_of(st.none(), st.text(), st.lists(st.text(), max_size=3))


# ---- mapping from Amazon's labels to our failure modes -------------------
@given(label=st.text(), sub=anything)
def test_mode_of_always_returns_a_known_mode(label, sub):
    assert mode_of(label, sub) in NAMES


@given(prefix=st.text(max_size=5), suffix=st.text(max_size=5), sub=anything)
def test_any_multi_label_is_multi_pick(prefix, suffix, sub):
    assert mode_of(prefix + "multi" + suffix, sub) == "multi_pick"


@given(sub=st.sampled_from(["open_box", "open_book", "book_jacket", "open_book_jacket", "partial_box"]))
def test_known_package_sublabels_map_to_themselves(sub):
    assert mode_of("package_defect", sub) == sub
    assert mode_of("package_defect", [sub]) == sub


# ---- invariants of the exported atlas.json -------------------------------
@settings(max_examples=300)
@given(p=st.sampled_from(ATLAS["points"]))
def test_every_point_is_valid(p):
    assert p["cluster"] in IDS
    assert 0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0


def test_cluster_counts_sum_to_points():
    assert sum(c["count"] for c in ATLAS["clusters"]) == len(ATLAS["points"])


@given(c=st.sampled_from(ATLAS["clusters"]))
def test_every_cluster_is_well_formed(c):
    assert 0 < c["share"] <= 1
    assert 1 <= len(c["examples"]) <= 10
    assert c["simulated"] == (c["id"] in SIMULATED)
    if c["simulated"]:
        assert len(c["scene_modes"]) == 3
