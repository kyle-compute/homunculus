# Google Scanned Objects: a mug, a plate and a toy

`ACE_Coffee_Mug_Kristen_16_oz_cup/`, `Ecoforms_Plant_Plate_S11Turquoise/` and
`Fisher_price_Classic_Toys_Buzzy_Bee/` are copied unchanged from
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects/tree/6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c/models),
commit 6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c.

- The meshes (`model.obj`) and textures (`texture.png`) are from Google Scanned Objects, by Google, under the
  [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/). Downie et al.,
  "Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items", ICRA 2022. The collision
  meshes (`model_collision_*.obj`) are convex pieces of the same scans, made for that repository.
- The MJCF files (`model.xml`) are by Kevin Zakka, under the MIT License in `LICENSE`.

`../bench.xml` places them on the bench with `<attach>`, which prefixes their names (`mug_`, `plate_`, `bee_`):
every model in that collection names its assets and body the same.
