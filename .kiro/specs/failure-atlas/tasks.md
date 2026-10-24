# Implementation plan: Failure Atlas (P1)

- [x] 1. Build the image list from ARMBench's per-image JSON labels (`00_make_manifest.py`) _Requirements: 1_
- [x] 2. Compute and cache CLIP embeddings (`01_embed.py`) _Requirements: 5_
- [x] 3. Try k-means clusters, review contact sheets, find they group by camera station _Requirements: 2 (revision note)_
- [x] 4. Define the 8 failure modes and the three scene recipes (`modes.py`) _Requirements: 2, 3_
- [x] 5. Group failures by mode and draw the 2D map (`02_cluster.py`) _Requirements: 2_
- [x] 6. Export `atlas.json`, thumbnails and results; train the classifier (`03_export.py`) _Requirements: 4, 5_
- [x] 7. Enforce the simulated rule in the export, with tests (`test_export.py`) _Requirements: 3_
- [x] 8. Validate `atlas.json` against the team contract (`validate_atlas.py`) _Requirements: 4_
- [x] 9. Property-based tests for the mapping and the exported file (`test_atlas.py`) _Requirements: 2, 3, 4_
- [x] 10. Build the Atlas page without external libraries (`p1/site/atlas.html`) _Requirements: 6_
- [x] 11. Add Kiro hooks for the tests and the contract check (`.kiro/hooks/p1-hooks.json`) _Requirements: 4_
- [x] 12. Add the P1 section to the README with the real numbers
- [ ] 13. (Not done) Check perception on OCID or OSD scenes
- [ ] 14. (Not done) Name modes with Bedrock captions
