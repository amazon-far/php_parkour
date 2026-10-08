# PHP training and evaluation

[Project overview](../README.md)

Train G1 terrain teachers and distill them into a RealSense D435i depth student.
Training and checkpoint evaluation require Linux/NVIDIA with IsaacSim.
For sim2sim with the exported ONNX policy in MuJoCo, see the [sim2sim guide](DEPLOY.md).

Run all commands below from the PHP checkout root.

## Setup

Setup uses the [Holosoma revision pinned by PHP](../README.md#get-started)
from `jinkunc/php-release-port`.

```bash
bash wbt_training/training_runs/setup_env.sh
source scripts/source_isaacsim_setup.sh
```

The default Conda environment is `php`. To use another name, export
`ENV_NAME=<name>` before setup and subsequent launches.

The training and evaluation launchers default to W&B logging (`LOGGER=wandb`).
Authenticate with your own account using `wandb login`. Set `LOGGER=disabled`
for local-only logging; checkpoints go under `logs/`. The Python training
presets also default to W&B; use `logger:disabled` when invoking Python directly.
Set `WANDB_BASE_URL` if you use a custom W&B server.

## Example motion/terrain datasets

Use these prepared datasets to **skip motion matching and start teacher training**.
They contain the motion/terrain pairs used by the current five-teacher training setup:
locomotion, low- and high-speed step, and low- and high-speed climb-76.
The motion files are already converted to the Holosoma training format at 50 FPS.
Each `*_motion.npz` has its matching `*_terrain.npy`; no generation, conversion,
or W&B dataset access is needed after download.

The bundles are available in the [training-motion-examples-v1 release](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1).

| Dataset | Download component | Motion/terrain pairs | Archive |
|---|---|---:|---:|
| Locomotion | `motions-locomotion` | 100 | [118.3 MB](https://github.com/amazon-far/php_parkour/releases/download/training-motion-examples-v1/php-motions-locomotion-v1.tar.gz) |
| Low-speed step | `motions-low-step` | 56 | [41.8 MB](https://github.com/amazon-far/php_parkour/releases/download/training-motion-examples-v1/php-motions-low-step-v1.tar.gz) |
| High-speed step | `motions-high-step` | 60 | [44.8 MB](https://github.com/amazon-far/php_parkour/releases/download/training-motion-examples-v1/php-motions-high-step-v1.tar.gz) |
| Low-speed climb-76 | `motions-low-climb-76` | 40 | [57.4 MB](https://github.com/amazon-far/php_parkour/releases/download/training-motion-examples-v1/php-motions-low-climb-76-v1.tar.gz) |
| High-speed climb-76 | `motions-high-climb-76` | 40 | [44.4 MB](https://github.com/amazon-far/php_parkour/releases/download/training-motion-examples-v1/php-motions-high-climb-76-v1.tar.gz) |

Download and extract a dataset from the PHP checkout root:

```bash
python scripts/download_assets.py motions-locomotion
```

Choose another download component from the table and use its directory in `REGISTRY`
to train that skill. The downloader verifies the archive and every extracted file against
[release-assets.json](../release-assets.json). Use `--destination DIR` for another
location. To verify a bundle already extracted above without network access:

```bash
python scripts/download_assets.py motions-locomotion --verify
```

These downloads provide training data, not pretrained teacher checkpoints.
[Train a teacher](#train-a-teacher) for each dataset, then [distill the teachers](#distill-a-student)
into a student. To generate different motions, follow the [motion-matching guide](../motion_matching/README.md).

## Train a teacher

After downloading an [example dataset](#example-motionterrain-datasets), point
`REGISTRY` at its directory:

```bash
REGISTRY="file://$HOME/.cache/php-parkour/motions-locomotion" \
LOGGER=disabled NGPUS=1 \
bash wbt_training/training_runs/run_terrain_teacher.sh \
    --training.headless True --training.num-envs 4096 \
    --training.name example-locomotion-teacher
```

Use `motions-low-step`, `motions-high-step`, `motions-low-climb-76`, or
`motions-high-climb-76` in both the download command and directory to train the
other skills. Train one teacher for each dataset you want the student to learn.
These bundles already contain the native Holosoma motion fields, so no conversion
or W&B dataset lookup is needed. `LOGGER=disabled` also disables W&B run logging;
omit it to use the default W&B logger after `wandb login`.

For data you prepared yourself, set `REGISTRY=file:///absolute/path/to/bundle`.
The directory must contain native Holosoma motion NPZ files and their matching
terrain NPY files. An explicit W&B artifact reference is also supported:

```bash
REGISTRY='your-team/your-project/your-motion-artifact:v0' \
NGPUS=1 bash wbt_training/training_runs/run_terrain_teacher.sh \
    --training.headless True --training.num-envs 4096 \
    --training.name my-teacher
```

To generate new datasets, follow the [motion-matching guide](../motion_matching/README.md).

For IsaacSim evaluation, use `CHECKPOINT=wandb://your-team/php-experiments/TEACHER_RUN_ID`
for the run's latest checkpoint, or `CHECKPOINT=/path/to/model_N.pt` for a local
checkpoint. See the [evaluation instructions](#evaluate-in-isaacsim).

## Distill a student

Student training defaults to `REGISTRY=auto` if no registry is supplied; it
recovers each W&B teacher's dataset. An explicitly set or exported `REGISTRY`
overrides this default.

After training a teacher, substitute its W&B run ID:

```bash
TEACHER_CHECKPOINT='wandb://your-team/php-experiments/TEACHER_RUN_ID' \
REGISTRY=auto NGPUS=1 \
bash wbt_training/training_runs/run_terrain_warp_distill.sh \
    --training.headless True --training.num-envs 4096 \
    --training.name my-student
```

For multiple teachers, pass comma-separated `TEACHER_CHECKPOINT` references;
`REGISTRY=auto` recovers one dataset per teacher in that order:

```bash
TEACHER_CHECKPOINT='wandb://your-team/php-experiments/CLIMB_RUN_ID,wandb://your-team/php-experiments/STEP_RUN_ID' \
REGISTRY=auto NGPUS=8 \
bash wbt_training/training_runs/run_terrain_warp_distill.sh \
    --training.headless True --training.num-envs 32768
```

## Launch options

Both teacher and student launchers support multiple GPUs and accept additional
configuration flags. Common controls are:

| Setting | Purpose |
|---|---|
| `NGPUS` / `CUDA_VISIBLE_DEVICES` | Number of training processes and selected GPU devices |
| `--training.num-envs` | **Total** parallel environments across GPUs |
| `RUN_NAME` | W&B display name when `LOGGER=wandb` |

## Evaluate and export your checkpoints

### Evaluate in IsaacSim

```bash
CHECKPOINT='wandb://your-team/php-experiments/TEACHER_RUN_ID' \
REGISTRY=auto bash wbt_training/training_runs/eval_teacher.sh

CHECKPOINT='wandb://your-team/php-experiments/STUDENT_RUN_ID' \
REGISTRY=auto bash wbt_training/training_runs/eval_student.sh
```

Run references select the latest checkpoint; append `/model_N.pt` to choose
one. Local checkpoints use `CHECKPOINT=/path/model.pt` with an explicit
`REGISTRY`. Supply registries explicitly if an older run lacks usable metadata.

Evaluation defaults to one environment on GPU 0. Set `NUM_ENVS`, `GPU_INDEX`,
or `MAX_EVAL_STEPS` as needed.
Add `--show-depth True` to student evaluation to view camera input.

### Export to ONNX

Student training automatically exports `depth_backbone.onnx` and `student.onnx`
after each checkpoint save and uploads them to the W&B run when logging is
enabled (the default). Use the pair from the latest `model_<iteration>/` folder
in the run's files.

To export a saved checkpoint manually:

```bash
python -m wbt_training.training_runs.export_distill_onnx \
    --training.checkpoint /path/to/student/model_29999.pt \
    --training.num-envs 1 logger:disabled
```

Manual export uses the saved registry references. For older checkpoints, add
`--training.registry-name` with the ordered datasets. Follow the
[sim2sim guide](DEPLOY.md) to run the exported pair in MuJoCo.
