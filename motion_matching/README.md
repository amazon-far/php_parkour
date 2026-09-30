# Motion matching

[Project overview](../README.md)

Generate parkour motion/terrain datasets for the Unitree G1.

## Setup

From the PHP checkout root, with Conda installed:

```bash
bash motion_matching/setup_env.sh
conda activate php-motion-matching
python scripts/download_assets.py databases
cd motion_matching
```

Run the remaining commands from `motion_matching/`. Databases default to
`~/.cache/php-parkour/databases`; set `PHP_MOTION_DATABASE_DIR` before downloading
and generating to choose another writable directory.

The downloader checks the archive and every extracted file against the SHA-256
manifest in [release-assets.json](../release-assets.json).

The download provides the motion-matching databases and source terrains.
Generation uses these to create the motion/terrain datasets for RL training.
Robot models and required meshes are included in the Python package.

## Generate and inspect

```bash
python -m motion_matching.run --mode generate --scenario high_speed_climb_76
python -m motion_matching.motion_visualizer \
    --motion-file motion_matching_results/high_speed_climb_76
```

Generated motions are exported at 50 FPS. Each motion has paired terrain files
in `motion_matching_results/<scenario>/`:

| File | Contents |
|---|---|
| `*_motion.npz` | Reference motion and per-frame direction commands, when supplied |
| `*_terrain.npy` | Box obstacles with randomized terrain variants |
| `*_terrain.obj` | Terrain mesh for visualization |

Choose a scenario from [builtin.py](motion_matching/scenarios/builtin.py), such
as `low_speed_step`, `high_speed_climb_76`, or `random_locomotion`. Open the
visualizer's URL to inspect a generated motion.

For interactive motion matching, run `python -m motion_matching.run`.
Interactive keyboard control needs Tkinter.

## Upload to a W&B registry

Create a W&B registry named `terrains-motions` that accepts `terrains-motions`
artifacts (or all types), then upload:

```bash
python -m pip install -e '.[registry]'
wandb login
python tracking_utils/upload_npz_newformat.py \
    --registry_name high-speed-climb-76 \
    --file_folder motion_matching_results/high_speed_climb_76 \
    --entity your-team
```

`--registry_name` names the collection. The uploader prints a `REGISTRY`
reference such as the one below; its organization may differ from your upload
team. You can select any available version or `:latest` for
[training](../wbt_training/README.md#train-a-teacher).

```bash
REGISTRY=your-org/wandb-registry-terrains-motions/high-speed-climb-76:v0
```

## Add your own motions or scenarios

For a 30 FPS G1 source clip (`qpos` shaped `(T, 36)` in
`[root_quat_wxyz, root_pos, dof29]` order), build a database with its terrain:

```bash
python -m motion_matching.generate_database \
    --motion-file /path/to/source.npz \
    --terrain-file /path/to/box.obj \
    --fps 30 --output database_g1_skill_name.bin
python -m motion_matching.play_database --database database_g1_skill_name.bin
```

Set `--fps` to the source clip's frame rate. The builder resamples the motion to
60 FPS and writes the database to the cache.

Register new skills in
[skill_registry.py](motion_matching/scenarios/skill_registry.py), compose
scenarios in [builtin.py](motion_matching/scenarios/builtin.py), and run them
with `--mode generate --scenario <name>`.
