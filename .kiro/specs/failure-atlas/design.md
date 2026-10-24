# Design: Failure Atlas (P1)

## Pipeline

```
ARMBench folder -> 00_make_manifest.py -> manifest.csv
  -> 01_embed.py (CLIP ViT-B/32) -> cache/embeddings.npy, cache/meta.csv
  -> 02_cluster.py (group by label, UMAP 2D map) -> cache/clusters.csv, review/*.jpg
  -> 03_export.py (classifier, thumbnails) -> p1/site/atlas.json, p1/results.md
  -> p1/site/atlas.html (reads atlas.json)
```

`modes.py` holds the single mapping from ARMBench label and sublabel to our 8 modes, and the three scene recipes for the simulated mode. Both `02_cluster.py` and `03_export.py` import it.

## Decisions

- **Labels, not k-means.** k-means on CLIP embeddings grouped photos by camera station (each station has a different background), so every cluster was about 50% multi-pick and 50% package defect. Amazon's sublabels are the real failure types.
- **One simulated mode.** Only multi-pick is a physical event a simulator can recreate. Package damage is shown as real data only.
- **Scene recipes are hypotheses.** The photos show the item on the gripper after the pick, not the tote beforehand, so the recipes (flat stacked, small beside big, against the wall) come from how suction grippers fail, not from the data.
- **Balanced sample.** 3,000 images per class keeps the embedding run short. Shares shown in the Atlas are shares of this sample, not real-world rates (the full dataset is about 40% multi-pick among failures).
- **No CDN.** The map is drawn on a canvas so the page works offline and when hosted.

## Interface: `atlas.json`

```
{"clusters": [{"id", "name", "defect", "count", "share", "simulated",
               "examples", "n_test", "detection_rate", "type_accuracy",
               "scene_modes"?}],
 "points":   [{"image", "x", "y", "cluster", "label"}],
 "classifier": {...}}
```

`validate_atlas.py` checks the contract keys; `check_simulated_count` checks the simulated rule.

## Error handling

- The manifest script prints raw labels so a bad mapping is visible.
- `03_export.py` fails loudly if the simulated rule is broken, and never leaves a half-written `atlas.json` (temp file, re-read, replace).
- The page shows the real error message if `atlas.json` cannot be loaded.

## Testing strategy

- Property-based tests (Hypothesis) in `test_atlas.py`: the label mapping always returns a known mode; any "multi" label is multi-pick; known sublabels map to themselves; every point has a real cluster and coordinates in [0, 1]; counts add up; exactly one simulated cluster with three scene modes.
- Unit tests in `test_export.py` for the simulated-count check.
- Hooks in `.kiro/hooks/p1-hooks.json` run the tests and the contract check when the agent edits P1 files.
