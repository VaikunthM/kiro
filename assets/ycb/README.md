# YCB mesh visuals

Textured meshes for the object types the scene generator uses, shown when `Sim(meshes=True)`. They are
visual only: collision, physics and outcomes always use the primitive box/cylinder of the scene's `size`.

| type | YCB object |
|---|---|
| sugar_box | 004 sugar box |
| tomato_soup_can | 005 tomato soup can |
| gelatin_box | 009 gelatin box |
| potted_meat_can | 010 potted meat can |
| foam_brick | 061 foam brick |

`flat_box` has no YCB counterpart and stays a plain box.

`manifest.json` holds, per type, the mesh `scale` and the visual geom `quat`/`pos` that fit the mesh into the
primitive's full extents. Regenerate it after changing a size with `python assets/ycb/make_manifest.py`.

## Credits

- Models: YCB Object and Model Set (Calli et al., ycbbenchmarks.com), Google 16k textured scans,
  licensed CC BY 4.0.
- Cleaned copies from github.com/elpis-lab/ycb_dataset (MIT License).
