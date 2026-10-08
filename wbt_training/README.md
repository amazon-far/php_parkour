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

## Train a teacher

Generate and upload a dataset using the
[motion-matching guide](../motion_matching/README.md#upload-to-a-wb-registry).
Choose a registry as the training dataset:

```bash
wandb login
export WANDB_ENTITY=your-team WANDB_PROJECT=php-experiments
export REGISTRY='your-org/wandb-registry-terrains-motions/high-speed-climb-76:latest'

NGPUS=1 \
bash wbt_training/training_runs/run_terrain_teacher.sh \
    --training.headless True --training.num-envs 4096 \
    --training.name my-teacher
```

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
