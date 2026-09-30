"""Export PHP depth-student ONNX bundles and Holosoma inference metadata.

The bundle contains a depth CNN and a student MLP. The student retains the
``obs`` / ``time_step`` inputs, actions, and six motion-reference outputs.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_to_csv(arr: list, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_onnx_metadata(onnx_path: str, metadata: dict[str, Any]) -> None:
    """Write key-value metadata into an existing ONNX file.

    Each value is stored as a comma-separated string (for lists) or a plain
    string, matching Holosoma inference's metadata format.
    """
    import onnx

    model = onnx.load(onnx_path)
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = _list_to_csv(value) if isinstance(value, list) else str(value)
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


# ---------------------------------------------------------------------------
# Depth-distillation exporter (student MLP + depth backbone)
# ---------------------------------------------------------------------------


class _OnnxDepthStudentPolicyExporter(torch.nn.Module):
    """Wrap a ``DepthStudentTeacher[Critic]`` student MLP for ONNX export.

    The depth backbone is exported separately via
    :func:`export_depth_backbone_as_onnx`. The student graph here takes the
    already-concatenated ``[actor_obs, depth_latent]`` tensor as ``obs``,
    matching the input shape the MLP was trained on (``num_actor_obs +
    depth_output_dim``).
    """

    def __init__(
        self, env, student_teacher_module, normalizer=None, verbose: bool = False
    ):
        super().__init__()
        self.verbose = verbose
        self._actor = copy.deepcopy(student_teacher_module.student)
        self.normalizer = (
            copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
        )

        # Embed the first reference clip, retaining the existing time_step
        # input and all six reference outputs for downstream consumers.
        motion = env.command_manager.get_state("motion_command").motion
        motion_ends = motion.motion_ends.to("cpu")
        first_end = torch.nonzero(motion_ends, as_tuple=False)
        end_idx = (
            first_end[0, 0].item() + 1
            if first_end.numel() > 0
            else motion.joint_pos.shape[0]
        )

        self.joint_pos = motion.joint_pos[:end_idx].to("cpu")
        self.joint_vel = motion.joint_vel[:end_idx].to("cpu")
        self.body_pos_w = motion.body_pos_w[:end_idx].to("cpu")
        self.body_quat_w = motion.body_quat_w[:end_idx].to("cpu")
        self.body_lin_vel_w = motion.body_lin_vel_w[:end_idx].to("cpu")
        self.body_ang_vel_w = motion.body_ang_vel_w[:end_idx].to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def in_features(self) -> int:
        return self._actor[0].in_features

    def forward(self, obs, time_step):
        ts = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        action = self._actor(self.normalizer(obs))
        return (
            action,
            self.joint_pos[ts],
            self.joint_vel[ts],
            self.body_pos_w[ts],
            self.body_quat_w[ts],
            self.body_lin_vel_w[ts],
            self.body_ang_vel_w[ts],
        )

    def export(self, path: str, filename: str = "student.onnx"):
        self.to("cpu")
        obs = torch.zeros(1, self.in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            dynamo=False,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def export_depth_backbone_as_onnx(
    depth_student_teacher,
    path: str,
    depth_shape: tuple[int, int],
    *,
    filename: str = "depth_backbone.onnx",
    verbose: bool = False,
) -> str:
    """Export the CNN depth backbone as a standalone ONNX file.

    The exported graph takes a 3D tensor ``(B, H, W)`` and returns a 2D
    latent ``(B, depth_output_dim)``. The backbone's ``forward`` unsqueezes
    the channel dim internally so the ONNX input stays 3D.
    """
    if not hasattr(depth_student_teacher, "depth_backbone"):
        raise ValueError("depth_student_teacher has no depth_backbone attribute")

    os.makedirs(path, exist_ok=True)
    # Export a copy so checkpoint saves preserve training mode and device.
    backbone = copy.deepcopy(depth_student_teacher.depth_backbone).cpu().eval()
    depth_input = torch.zeros(1, *depth_shape)
    torch.onnx.export(
        backbone,
        (depth_input,),
        os.path.join(path, filename),
        export_params=True,
        opset_version=11,
        dynamo=False,
        verbose=verbose,
        input_names=["depth_image"],
        output_names=["depth_latent"],
        dynamic_axes={},
    )

    return os.path.join(path, filename)


def export_depth_student_as_onnx(
    env,
    depth_student_teacher,
    path: str,
    *,
    normalizer=None,
    filename: str = "student.onnx",
    verbose: bool = False,
) -> str:
    """Export the student MLP (proprio + depth_latent -> actions) as ONNX."""
    os.makedirs(path, exist_ok=True)
    exporter = _OnnxDepthStudentPolicyExporter(
        env, depth_student_teacher, normalizer, verbose
    )
    exporter.export(path, filename)
    return os.path.join(path, filename)


def export_depth_student_components(
    env,
    depth_student_teacher,
    path: str,
    depth_shape: tuple[int, int],
    *,
    normalizer=None,
    depth_backbone_filename: str = "depth_backbone.onnx",
    student_filename: str = "student.onnx",
    verbose: bool = False,
) -> tuple[str, str]:
    """Export both depth_backbone.onnx and student.onnx to ``path``.

    Returns ``(backbone_path, student_path)``.
    """
    os.makedirs(path, exist_ok=True)
    backbone_path = export_depth_backbone_as_onnx(
        depth_student_teacher,
        path,
        depth_shape,
        filename=depth_backbone_filename,
        verbose=verbose,
    )
    student_path = export_depth_student_as_onnx(
        env,
        depth_student_teacher,
        path,
        normalizer=normalizer,
        filename=student_filename,
        verbose=verbose,
    )
    return backbone_path, student_path


# ---------------------------------------------------------------------------


def export_depth_student_bundle(algo, iteration: int) -> str | None:
    """Export and annotate a deployable depth-student bundle for a checkpoint."""
    if not getattr(algo, "is_main_process", True):
        return None

    output_dir = os.path.join(algo.log_dir, f"model_{iteration:05d}")
    backbone_path, student_path = export_depth_student_components(
        env=algo.env,
        depth_student_teacher=algo.policy,
        path=output_dir,
        depth_shape=algo.depth_shape,
        normalizer=None,
    )

    try:
        import wandb
    except ImportError:
        wandb = None

    run_path = ""
    if wandb is not None and wandb.run is not None:
        run_path = wandb.run.path or ""

    metadata = build_metadata_from_wbt_env(algo.env, run_path=run_path)
    attach_onnx_metadata(student_path, metadata)
    attach_onnx_metadata(backbone_path, {"run_path": run_path})

    if wandb is not None and wandb.run is not None:
        wandb.save(student_path, base_path=algo.log_dir)
        wandb.save(backbone_path, base_path=algo.log_dir)
    return output_dir


# Metadata for Holosoma whole-body tracking environments.


def build_metadata_from_wbt_env(env, run_path: str = "") -> dict[str, Any]:
    """Extract ONNX metadata from a holosoma ``WholeBodyTrackingManager`` env.

    Emits both ``joint_stiffness``/``joint_damping`` and ``kp``/``kd`` for
    inference readers using either spelling, pointing at the same
    values. Fields that aren't available on the env are omitted silently;
    the three critical keys (``joint_names``, ``action_scale``,
    ``default_joint_pos``) raise a clear error instead.
    """
    metadata: dict[str, Any] = {"run_path": run_path}

    # Critical: joint_names (drives inference joint reordering)
    try:
        metadata["joint_names"] = list(env.robot_config.dof_names)
    except AttributeError as e:
        raise RuntimeError(f"env.robot_config.dof_names not available: {e}") from e

    # The distill preset registers the action under "joint_control"
    # (config_values/terrain.py:309-318); fall back to the first registered
    # term so this keeps working if upstream renames it.
    action_term = None
    for candidate in ("joint_control", "joint_pos"):
        try:
            action_term = env.action_manager.get_term(candidate)
            break
        except KeyError:
            continue
    if action_term is None:
        names = list(getattr(env.action_manager, "_term_instances", {}).keys())
        if names:
            action_term = env.action_manager.get_term(names[0])
        else:
            raise RuntimeError("env.action_manager has no registered action terms")

    # Critical: action_scale (holosoma stores a per-joint tensor; emit as list)
    try:
        metadata["action_scale"] = action_term.action_scales.cpu().tolist()
    except AttributeError as e:
        raise RuntimeError(f"action_term.action_scales not available: {e}") from e

    # Critical: default_joint_pos (used by the control-gain fallback)
    try:
        # Prefer the non-randomised canonical pose if available. Both
        # `default_dof_pos_base` (1D) and `default_dof_pos` (2D, per-env)
        # exist; flatten so the CSV writer formats floats instead of falling
        # through to ``str(inner_list)``.
        if hasattr(env, "default_dof_pos_base"):
            flat = env.default_dof_pos_base.flatten().cpu().tolist()
        else:
            flat = env.default_dof_pos[0].flatten().cpu().tolist()
        metadata["default_joint_pos"] = flat
    except AttributeError as e:
        raise RuntimeError(f"env.default_dof_pos not available: {e}") from e

    # Optional: control gains. ``JointPositionTargetActionTerm`` (implicit PD)
    # doesn't expose ``p_gains``/``d_gains`` because the gains live on the
    # articulation drive, not the action term — ``robot_config.control``
    # holds them in a dict keyed by joint *group* (e.g. ``hip_yaw`` matches
    # ``left_hip_yaw_joint``). Resolve per joint using substring match, the
    # same convention the action term uses to derive ``action_scales``
    # (joint_control.py:46-53).
    p_gains = d_gains = None
    if hasattr(action_term, "p_gains"):
        try:
            p_gains = action_term.p_gains.cpu().tolist()
            d_gains = action_term.d_gains.cpu().tolist()
        except AttributeError:
            p_gains = d_gains = None
    if p_gains is None:
        try:
            stiffness = env.robot_config.control.stiffness
            damping = env.robot_config.control.damping
            dof_names = list(env.robot_config.dof_names)

            def _resolve(d, name):
                if not isinstance(d, dict):
                    return (
                        float(d[dof_names.index(name)])
                        if hasattr(d, "__getitem__")
                        else 0.0
                    )
                stripped = name.replace("_joint", "")
                for k, v in d.items():
                    if k in stripped:
                        return float(v)
                return 0.0

            p_gains = [_resolve(stiffness, n) for n in dof_names]
            d_gains = [_resolve(damping, n) for n in dof_names]
        except AttributeError:
            p_gains = d_gains = None
    if p_gains is not None:
        metadata["joint_stiffness"] = p_gains
        metadata["joint_damping"] = d_gains
        metadata["kp"] = p_gains
        metadata["kd"] = d_gains

    # Optional: command/observation term names
    try:
        metadata["command_names"] = list(env.command_manager.cfg.setup_terms.keys())
    except AttributeError:
        pass

    try:
        # ObservationManager concatenates sorted term names, then the policy
        # appends the depth latent. Keep the established latent-slot label.
        actor_terms = sorted(env.observation_manager.cfg.groups["actor_obs"].terms)
        metadata["observation_names"] = actor_terms + ["placeholder"]
    except (AttributeError, KeyError):
        pass

    # Optional: motion anchor + tracked body names
    try:
        motion_cmd = env.command_manager.get_state("motion_command")
        motion_cfg = motion_cmd.motion_cfg
        metadata["anchor_body_name"] = motion_cfg.body_name_ref[0]
        metadata["body_names"] = list(motion_cfg.body_names_to_track)
    except (AttributeError, IndexError):
        pass

    return metadata
