# PHP sim2sim with the RealSense D435i student

[Project overview](../README.md)

Run the exported PHP depth policy in MuJoCo with the PHP-owned
[run_php_sim.sh](../run_php_sim.sh) and
[run_php_inference.sh](../run_php_inference.sh) launchers. They select the
RealSense D435i simulation and inference presets from this checkout's pinned
[Holosoma submodule](../README.md#get-started).
A student consists of two local files, `depth_backbone.onnx` and `student.onnx`.
The depth backbone encodes the camera image; the student combines that latent
with proprioception and a direction command to produce joint targets.

For policy downloads, see [Example results](../README.md#example-results).

## Get Holosoma and install the runtimes

The wrappers activate `hsmujoco` for simulation and `hsinference` for inference,
under `~/.holosoma_deps/miniconda3` by default. Initialize the pinned submodule;
skip the two setup commands if these environments are already installed:

```bash
git submodule update --init --recursive thirdparty/holosoma
bash thirdparty/holosoma/scripts/setup_mujoco.sh
bash thirdparty/holosoma/scripts/setup_inference.sh
```

If reusing environments installed from another checkout, install these packages
into them so imports resolve here. This replaces their existing editable installs:

```bash
"$HOME/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python" \
    -m pip install --no-deps -e ./thirdparty/holosoma/src/holosoma
"$HOME/.holosoma_deps/miniconda3/envs/hsinference/bin/python" \
    -m pip install --no-deps -e ./thirdparty/holosoma/src/holosoma_inference
```

See the [Holosoma inference guide](https://github.com/amazon-far/holosoma/blob/deeab8038ab76097a23fb90c1f6a31914c49f883/src/holosoma_inference/README.md)
for supported systems and runtime dependencies. This standalone route uses
Holosoma's local simulator bridge and depth shared memory; it does not require
FAR-pi.

## Start simulation and inference

Run both launchers from the PHP checkout on the same machine. They use
`thirdparty/holosoma` by default; set `HOLOSOMA_ROOT=/path/to/holosoma` to test
against another compatible checkout. Use a graphical desktop, or set `DISPLAY`
in both terminals to that desktop's X server when connecting over SSH. Without
a display, the simulator's gantry keys are unavailable and policy direction
keys cannot detect releases.

Stop any previous PHP simulator and inference processes before restarting:
the simulator wrapper recreates `depth_img_shm`.

Start the simulator first:

```bash
# Terminal 1
bash ./run_php_sim.sh
```

The wrapper selects the RealSense D435i camera and depth publisher, the
half-sphere-hand G1 model, and the local simulator bridge. Wait for:

```text
[DepthShmPlugin] created 'depth_img_shm' (20184 bytes) shape=(1, 1, 58, 87)
```

The default stepped course is bundled with Holosoma. Append
`terrain:terrain-locomotion-plane` for flat ground, or
`--terrain.terrain-term.obj-file-path /absolute/path/to/course.obj` for a custom
mesh. Other CLI overrides are also forwarded by the wrapper.

Then start inference with a matching ONNX pair:

```bash
# Terminal 2
PHP_STUDENT_DIR="$HOME/.cache/php-parkour/student"
BACKBONE="$PHP_STUDENT_DIR/depth_backbone.onnx" \
STUDENT="$PHP_STUDENT_DIR/student.onnx" \
bash ./run_php_inference.sh
```

Change `PHP_STUDENT_DIR` to use the pair from your own training run. The wrapper
passes the backbone and student together, in that order, and selects the local
interface `lo`. Local ONNX files require no W&B access or training registry.
Always supply both paths or your own `RUN` and `STEP` rather than relying on
the script's built-in model defaults.

Alternatively, load an already-exported pair from W&B:

```bash
RUN='wandb://your-team/php-experiments/STUDENT_RUN_ID' \
STEP=model_20000 \
bash ./run_php_inference.sh
```

Set `STEP` to an ONNX folder in the run's files. This published wrapper
does not resolve `STEP=latest` or convert `.pt` checkpoints automatically;
use the [export instructions](README.md#export-to-onnx) if needed.

Before starting the policy, confirm that inference reports:

```text
[DepthShmSensor] attached to 'depth_img_shm' shape=(1, 1, 58, 87)
```


## Control the robot

Lower the gantry with `8` in the MuJoCo window until the feet reach the ground,
then remove it with `9`. Press `]` in the policy terminal to start the policy.
`o` enters damping mode; `i` returns to the stiff startup pose. These are the
native inference controls.
Press `Backspace` in the MuJoCo window to reset the environment.

| Direction command | Policy-terminal keys |
|---|---|
| Forward / backward | `w` / `s` |
| 45° left / right | `a` / `d` |
| 90° left / right | `q` / `e` |
| Toggle speed mode | `=` |

The [Sim-to-Sim Depth Locomotion workflow](https://github.com/amazon-far/holosoma/blob/deeab8038ab76097a23fb90c1f6a31914c49f883/src/holosoma_inference/docs/workflows/sim-to-sim-depth-locomotion.md)
shows the underlying Holosoma commands, depth image configuration, and
keyboard controls.
