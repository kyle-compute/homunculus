# Google Scanned Objects: a mug and a bowl

`Cole_Hardware_Mug_Classic_Blue/` and `Cole_Hardware_Bowl_Scirocco_YellowBlue/` are copied unchanged from
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects/tree/6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c/models),
commit 6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c.

- The meshes (`model.obj`) and textures (`texture.png`) are from Google Scanned Objects, by Google, under the
  [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/). Downie et al.,
  "Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items", ICRA 2022. The collision
  meshes (`model_collision_*.obj`) are convex pieces of the same scans, made for that repository.
- The MJCF files (`model.xml`) are by Kevin Zakka, under the MIT License in `LICENSE`.

`../table.xml` places them on the table with `<attach>`, which prefixes their names
(`mug_`, `bowl_`): every model in that collection names its assets and body the same.
