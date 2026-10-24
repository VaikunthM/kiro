from sim.validate_scenes import check_scene

GOOD = {"scene_id": "ok", "objects": [
    {"id": "a", "shape": "box", "size": [0.1, 0.08, 0.04], "pos": [0, 0, 0.02]}]}


def test_good_scene_has_no_problems():
    assert check_scene(GOOD) == []


def test_overlapping_objects_are_reported():
    bad = {"scene_id": "overlap", "objects": [
        {"id": "a", "shape": "box", "size": [0.1, 0.08, 0.04], "pos": [0, 0, 0.02]},
        {"id": "b", "shape": "box", "size": [0.1, 0.08, 0.04], "pos": [0.03, 0, 0.02]}]}
    problems = check_scene(bad)
    assert any("moved" in p for p in problems)


def test_object_outside_tote_is_reported():
    bad = {"scene_id": "outside", "objects": [
        {"id": "a", "shape": "box", "size": [0.1, 0.08, 0.04], "pos": [0.5, 0, 0.02]}]}
    assert any("outside" in p for p in check_scene(bad))


def test_missing_field_is_reported_not_raised():
    bad = {"scene_id": "nofield", "objects": [{"id": "a", "shape": "box", "pos": [0, 0, 0.02]}]}
    assert any("size" in p for p in check_scene(bad))
