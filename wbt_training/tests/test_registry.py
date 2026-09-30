"""Registry provenance, reproducible inputs, and dataset pairing without W&B."""

from __future__ import annotations

import dataclasses
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wbt_training.utils import multi_motion_helpers
from wbt_training.utils.registry import (
    ARTIFACT_TYPE,
    METADATA_KEY,
    RegistryResolver,
    ResolvedRegistry,
    bundle_pairs,
    prepare_inputs,
    record_registry_usage,
)

COLLECTION = "public-org/wandb-registry-terrains-motions/stairs"
PINNED = COLLECTION + ":v7"
OTHER = "public-org/wandb-registry-terrains-motions/flat:v2"


def unexpected_network(*args, **kwargs):
    pytest.fail(f"Unexpected W&B service call: {args!r} {kwargs!r}")


class FakeConfig(dict):
    def update(self, values, *, allow_val_change=False):
        self.allow_val_change = allow_val_change
        super().update(values)


class FakeRun:
    def __init__(self, config=None, artifacts=()):
        self.config = FakeConfig(config or {})
        self.artifacts = list(artifacts)
        self.used_artifacts_calls = 0
        self.use_calls = []
        self.link_artifact = unexpected_network

    def used_artifacts(self):
        self.used_artifacts_calls += 1
        return self.artifacts

    def use_artifact(self, name, *, type):
        self.use_calls.append((name, type))


class FakeArtifact:
    def __init__(
        self, source, qualified_name=PINNED, *, version=None, kind=ARTIFACT_TYPE
    ):
        self.source = source
        self.qualified_name = qualified_name
        self.version = version or qualified_name.rsplit(":", 1)[-1]
        self.type = kind
        self.download_calls = []

    def download(self, *, root):
        self.download_calls.append(Path(root))
        shutil.copytree(self.source, root)
        return root


class FakeApi:
    def __init__(self, artifacts=None, runs=None):
        self.artifacts = artifacts or {}
        self.runs = runs or {}
        self.artifact_calls = []
        self.run_calls = []

    def artifact(self, reference):
        self.artifact_calls.append(reference)
        assert reference in self.artifacts, f"Unexpected artifact: {reference}"
        return self.artifacts[reference]

    def run(self, path):
        self.run_calls.append(path)
        assert path in self.runs, f"Unexpected run: {path}"
        return self.runs[path]


@pytest.fixture(autouse=True)
def isolated_services(tmp_path, monkeypatch):
    """No test can accidentally use the developer's endpoint, cache, or account."""
    monkeypatch.setenv("CONVERT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    fake = SimpleNamespace(
        Api=unexpected_network,
        init=unexpected_network,
        setup=unexpected_network,
        Artifact=unexpected_network,
        run=None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    scratch = tmp_path / "combined"
    scratch.mkdir()
    monkeypatch.setattr(multi_motion_helpers, "_TMP_DIR", scratch)
    return fake


def write_native(path, value=1, *, frames=3):
    """Small valid native fixtures intentionally do not assume a 29-DOF robot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fps": np.array([50]),
        "joint_pos": np.full((frames, 2), value, dtype=np.float32),
        "joint_vel": np.zeros((frames, 2), dtype=np.float32),
        "body_pos_w": np.tile([[[value, 0, 0.7]]], (frames, 1, 1)),
        "body_quat_w": np.tile([[[1, 0, 0, 0]]], (frames, 1, 1)),
        "body_lin_vel_w": np.zeros((frames, 1, 3)),
        "body_ang_vel_w": np.zeros((frames, 1, 3)),
        "joint_names": np.array(["left", "right"]),
        "body_names": np.array(["pelvis"]),
        "vel_cmd": np.full((frames, 15), value, dtype=np.float32),
    }
    np.savez(path, **data)
    return data


def write_terrain(path, value=1, *, boxes=1):
    data = np.tile([[[value, 0, 0.3, 1, 0, 0, 0, 0.5, 0.8, 0.6]]], (boxes, 2, 1))
    np.save(path, data)
    return data


@pytest.fixture
def native_bundle(tmp_path):
    root = tmp_path / "native"
    write_native(root / "sample_motion.npz")
    write_terrain(root / "sample_terrain.npy")
    (root / "sample_terrain.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return root


@pytest.mark.parametrize(
    "requested", [COLLECTION, COLLECTION + ":latest", "wandb://" + COLLECTION + ":best"]
)
def test_alias_pins_source_reference_separately_from_native_cache(
    native_bundle, requested
):
    normalized = requested.removeprefix("wandb://")
    if ":" not in normalized:
        normalized += ":latest"
    artifact = FakeArtifact(native_bundle)
    api = FakeApi({normalized: artifact})
    resolver = RegistryResolver(api=api)

    (result,) = resolver.resolve(requested)

    assert result.reference == PINNED
    assert result.directory != native_bundle
    assert result.local_uri == result.directory.as_uri()
    assert result.reference != result.local_uri
    for original in native_bundle.iterdir():
        assert (result.directory / original.name).read_bytes() == original.read_bytes()
    # Aliases are checked again, but the immutable data is downloaded only once.
    assert resolver.resolve(requested) == [result]
    assert api.artifact_calls == [normalized, normalized]
    assert len(artifact.download_calls) == 1


def test_custom_registry_with_spaces_survives_pinning_recording_and_auto_recovery(
    native_bundle, isolated_services
):
    collection = "public-org/wandb-registry-PHP Motion Datasets/stairs"
    alias = collection + ":latest"
    pinned = collection + ":v7"
    api = FakeApi({alias: FakeArtifact(native_bundle, pinned)})
    resolver = RegistryResolver(api=api)
    _, results = prepare_inputs("teacher", "", "wandb://" + alias, resolver=resolver)
    assert [r.reference for r in results] == [pinned]

    run = FakeRun()
    isolated_services.run = run
    record_registry_usage(results, "wandb")
    assert run.use_calls == [(pinned, ARTIFACT_TYPE)]
    assert run.config[METADATA_KEY]["registry_names"] == [pinned]
    assert run.config["training"]["registry_name"] == pinned

    api.runs["team/project/student"] = run
    _, recovered = prepare_inputs(
        "eval_student", "wandb://team/project/student", "auto", resolver=resolver
    )
    assert recovered == results
    assert api.artifact_calls == [alias]
    assert run.used_artifacts_calls == 0


@pytest.mark.parametrize("requested", [COLLECTION + ":latest", PINNED])
def test_linked_qualified_name_version_wins_over_source_artifact_version(
    native_bundle, requested
):
    # W&B collection v7 may link to source-project artifact v2.
    artifact = FakeArtifact(native_bundle, PINNED, version="v2")
    (result,) = RegistryResolver(api=FakeApi({requested: artifact})).resolve(requested)
    assert result.reference == PINNED
    assert (result.directory / "sample_motion.npz").is_file()


def test_requested_immutable_version_must_match_qualified_name(native_bundle):
    artifact = FakeArtifact(native_bundle, COLLECTION + ":v8", version="v7")
    with pytest.raises(ValueError, match="version|Requested|requested"):
        RegistryResolver(api=FakeApi({PINNED: artifact})).resolve(PINNED)
    assert artifact.download_calls == []


def test_immutable_input_reuses_cache_offline_and_redownloads_after_removal(
    native_bundle,
):
    artifact = FakeArtifact(native_bundle)
    api = FakeApi({PINNED: artifact})
    (result,) = RegistryResolver(api=api).resolve(PINNED)
    before = (result.directory / "sample_motion.npz").read_bytes()

    # This fresh resolver has no API; the global W&B API factory is forbidden.
    assert RegistryResolver().resolve(PINNED) == [result]
    shutil.rmtree(result.directory)
    (regenerated,) = RegistryResolver(api=api).resolve(PINNED)

    assert regenerated == result
    assert (regenerated.directory / "sample_motion.npz").read_bytes() == before
    assert len(artifact.download_calls) == 2
    assert api.artifact_calls == [PINNED, PINNED]


def test_alias_moves_to_new_version_without_reusing_old_payload(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_native(first / "motion.npz", 1)
    write_native(second / "motion.npz", 9)
    alias = COLLECTION + ":latest"
    api = FakeApi({alias: FakeArtifact(first)})
    resolver = RegistryResolver(api=api)
    (old,) = resolver.resolve(alias)
    api.artifacts[alias] = FakeArtifact(second, COLLECTION + ":v8")
    (new,) = resolver.resolve(alias)

    assert old.reference == PINNED
    assert new.reference == COLLECTION + ":v8"
    assert old.directory != new.directory
    with np.load(old.directory / "motion.npz") as data:
        np.testing.assert_array_equal(data["joint_pos"], 1)
    with np.load(new.directory / "motion.npz") as data:
        np.testing.assert_array_equal(data["joint_pos"], 9)


def test_cache_is_scoped_to_wandb_server(native_bundle, monkeypatch):
    artifact = FakeArtifact(native_bundle)
    resolver = RegistryResolver(api=FakeApi({PINNED: artifact}))
    (public,) = resolver.resolve(PINNED)
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.example.test")
    (separate,) = resolver.resolve(PINNED)
    assert separate.reference == public.reference
    assert separate.directory != public.directory
    assert len(artifact.download_calls) == 2


def test_beyondmimic_conversion_preserves_commands_and_welded_feet(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    frames = 4
    data = {
        "fps": np.array([50]),
        "joint_pos": np.arange(frames * 29).reshape(frames, 29).astype(np.float32),
        "joint_vel": np.arange(frames * 29).reshape(frames, 29).astype(np.float32) / 10,
        "body_pos_w": np.arange(frames * 30 * 3)
        .reshape(frames, 30, 3)
        .astype(np.float32),
        "body_quat_w": np.tile([[[1, 0, 0, 0]]], (frames, 30, 1)),
        "body_lin_vel_w": np.full((frames, 30, 3), 2.0),
        "body_ang_vel_w": np.full((frames, 30, 3), 3.0),
        "vel_cmd": np.arange(frames * 15).reshape(frames, 15).astype(np.float32),
        "motion_ends": np.array([False, True, False, True]),
        "motion_idxs": np.array([0, 0, 1, 1]),
    }
    motion = root / "sample_motion.npz"
    np.savez(motion, **data)
    original = motion.read_bytes()
    terrain = write_terrain(root / "sample_terrain.npy")
    (result,) = RegistryResolver(api=FakeApi({PINNED: FakeArtifact(root)})).resolve(
        PINNED
    )

    with np.load(result.directory / motion.name) as converted:
        assert converted["joint_pos"].shape == (frames, 36)
        assert converted["joint_vel"].shape == (frames, 35)
        assert converted["body_pos_w"].shape == (frames, 32, 3)
        for key in ("fps", "vel_cmd", "motion_ends", "motion_idxs"):
            np.testing.assert_array_equal(converted[key], data[key])
        np.testing.assert_array_equal(
            converted["joint_pos"][:, :3], data["body_pos_w"][:, 0]
        )
        np.testing.assert_array_equal(
            converted["joint_pos"][:, 3:7], data["body_quat_w"][:, 0]
        )
        names = converted["body_names"].tolist()
        for side in ("left", "right"):
            ankle = names.index(f"{side}_ankle_roll_link")
            foot = names.index(f"{side}_foot_contact_point")
            for key in (
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                np.testing.assert_array_equal(
                    converted[key][:, foot], converted[key][:, ankle]
                )
    np.testing.assert_array_equal(
        np.load(result.directory / "sample_terrain.npy"), terrain
    )
    assert motion.read_bytes() == original


def test_recursive_duplicate_basenames_are_paired_and_copied_in_path_order(tmp_path):
    root = tmp_path / "source"
    # Reverse creation order; repeated basenames cannot be flattened safely.
    for name, value in (("zeta", 9), ("alpha", 1)):
        write_native(root / name / "sample_motion.npz", value)
        write_terrain(root / name / "sample_terrain.npy", value)
        (root / name / "sample_terrain.obj").write_text(f"# {name}\n")
    pairs = bundle_pairs(root)
    assert [
        (m.relative_to(root).as_posix(), t.relative_to(root).as_posix())
        for m, t, _ in pairs
    ] == [
        ("alpha/sample_motion.npz", "alpha/sample_terrain.npy"),
        ("zeta/sample_motion.npz", "zeta/sample_terrain.npy"),
    ]
    (result,) = RegistryResolver(api=FakeApi({PINNED: FakeArtifact(root)})).resolve(
        PINNED
    )
    for original in root.rglob("*"):
        if original.is_file():
            assert (
                result.directory / original.relative_to(root)
            ).read_bytes() == original.read_bytes()


@pytest.mark.parametrize(
    "order", [("flat", "stairs"), ("stairs", "flat"), ("flat", "stairs", "flat")]
)
def test_terrain_boundaries_include_every_motion_only_clip(tmp_path, order):
    flat, stairs = tmp_path / "flat", tmp_path / "stairs"
    write_native(flat / "sample_motion.npz", 3, frames=2)
    write_native(stairs / "z" / "sample_motion.npz", 9, frames=4)
    write_terrain(stairs / "z" / "sample_terrain.npy", 9, boxes=2)
    write_native(stairs / "a" / "sample_motion.npz", 1, frames=3)
    write_terrain(stairs / "a" / "sample_terrain.npy", 1)
    refs = [(tmp_path / name).as_uri() for name in order]

    counts, motion_path, terrain_path, _ = (
        multi_motion_helpers.pull_paired_from_wandb_registry(refs)
    )

    assert counts == [1 if name == "flat" else 2 for name in order]
    values, indices, ends, boundaries, boxes = [], [], [], [0], []
    for index, name in enumerate(order):
        clips = [(3, 2, 0)] if name == "flat" else [(1, 3, 1), (9, 4, 2)]
        for value, frames, box_count in clips:
            values.extend([value] * frames)
            indices.extend([index] * frames)
            ends.extend([False] * (frames - 1) + [True])
            boundaries.append(boundaries[-1] + box_count)
            boxes.extend([value] * box_count)
    with np.load(motion_path) as motion:
        np.testing.assert_array_equal(motion["joint_pos"][:, 0], values)
        np.testing.assert_array_equal(motion["vel_cmd"][:, 0], values)
        np.testing.assert_array_equal(motion["motion_idxs"], indices)
        np.testing.assert_array_equal(motion["motion_ends"], ends)
    with np.load(terrain_path) as terrain:
        np.testing.assert_array_equal(terrain["obj_count_list"], boundaries)
        np.testing.assert_array_equal(terrain["obj_list"][:, 0, 0], boxes)


@pytest.mark.parametrize("bad_layout", ["missing-pair", "orphan", "mesh-npz"])
def test_pair_errors_are_reported_before_network_requests(tmp_path, bad_layout):
    root = tmp_path / "bad"
    write_native(root / "a_motion.npz")
    write_terrain(root / "a_terrain.npy")
    if bad_layout == "missing-pair":
        write_native(root / "b_motion.npz")
    elif bad_layout == "orphan":
        write_terrain(root / "b_terrain.npy")
    else:
        np.savez(root / "legacy_terrain.npz", vertices=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="pair|terrain NPZ"):
        # The remote entry intentionally precedes the bad local one.
        RegistryResolver().resolve([PINNED, root.as_uri()])


def test_local_inputs_remain_offline_and_keep_their_identity(native_bundle):
    checkpoints, results = prepare_inputs("teacher", "", native_bundle.as_uri())
    assert checkpoints == []
    assert results == [ResolvedRegistry(native_bundle.as_uri(), native_bundle)]


@pytest.mark.parametrize("mode", ["eval_student", "eval_teacher"])
def test_ordered_metadata_precedes_legacy_and_used_artifacts_and_keeps_duplicates(
    native_bundle, mode
):
    names = [OTHER, PINNED, OTHER]
    run = FakeRun(
        {
            METADATA_KEY: {"version": 1, "registry_names": names},
            "training": {"registry_name": PINNED},
        },
        [FakeArtifact(native_bundle, PINNED)],
    )
    api = FakeApi(
        {ref: FakeArtifact(native_bundle, ref) for ref in set(names)},
        {"team/project/student": run},
    )
    checkpoints, results = prepare_inputs(
        mode,
        "wandb://team/project/runs/student/model_42.pt",
        "auto",
        resolver=RegistryResolver(api=api),
    )
    assert checkpoints == ["wandb://team/project/student/model_42.pt"]
    assert [r.reference for r in results] == names
    assert results[0].directory == results[2].directory
    assert api.run_calls == ["team/project/student"]
    assert api.artifact_calls == [OTHER, PINNED]
    assert run.used_artifacts_calls == 0


def test_multiteacher_auto_follows_checkpoint_order(native_bundle):
    api = FakeApi(
        {ref: FakeArtifact(native_bundle, ref) for ref in (OTHER, PINNED)},
        {
            "team/project/z_teacher": FakeRun(
                {METADATA_KEY: {"version": 1, "registry_names": [OTHER]}}
            ),
            "team/project/a_teacher": FakeRun(
                {METADATA_KEY: {"version": 1, "registry_names": [PINNED]}}
            ),
        },
    )
    checkpoints, results = prepare_inputs(
        "distill",
        "wandb://team/project/z_teacher/model_9.pt,wandb://team/project/a_teacher",
        "auto",
        resolver=RegistryResolver(api=api),
    )
    assert checkpoints == [
        "wandb://team/project/z_teacher/model_9.pt",
        "wandb://team/project/a_teacher",
    ]
    assert [r.reference for r in results] == [OTHER, PINNED]
    assert api.run_calls == ["team/project/z_teacher", "team/project/a_teacher"]


def test_distillation_rejects_teacher_with_multiple_registry_inputs():
    run = FakeRun({METADATA_KEY: {"version": 1, "registry_names": [PINNED, OTHER]}})
    api = FakeApi(runs={"team/project/teacher": run})
    with pytest.raises(ValueError, match="one REGISTRY per TEACHER_CHECKPOINT"):
        prepare_inputs(
            "distill",
            "wandb://team/project/teacher",
            "auto",
            resolver=RegistryResolver(api=api),
        )
    assert api.artifact_calls == []


def test_legacy_saved_pinned_references_retain_order_without_artifact_lookup():
    run = FakeRun({"training": {"registry_name": f"{OTHER},{PINNED},{OTHER}"}})
    resolver = RegistryResolver(api=FakeApi(runs={"team/project/student": run}))
    assert resolver.for_run("wandb://team/project/student") == [OTHER, PINNED, OTHER]
    assert run.used_artifacts_calls == 0


@pytest.mark.parametrize(
    "stored", [None, COLLECTION + ":latest", "file:///missing/php-conversion-cache"]
)
def test_legacy_single_dataset_artifact_recovers_missing_or_mutable_config(
    native_bundle, stored
):
    run = FakeRun(
        {"training": {"registry_name": stored}},
        [
            FakeArtifact(native_bundle, "team/project/model:v1", kind="model"),
            FakeArtifact(native_bundle),
        ],
    )
    resolver = RegistryResolver(api=FakeApi(runs={"team/project/teacher": run}))
    assert resolver.for_run("wandb://team/project/teacher") == [PINNED]
    assert run.used_artifacts_calls == 1


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {"version": 2, "registry_names": [PINNED]},
        {"version": 1, "registry_names": []},
        {"version": 1, "registry_names": PINNED},
        {"version": 1, "registry_names": [None]},
        {"version": 1, "registry_names": [COLLECTION + ":latest"]},
        {"version": 1, "registry_names": ["malformed"]},
    ],
)
def test_malformed_ordered_metadata_is_not_silently_overridden(native_bundle, metadata):
    run = FakeRun(
        {METADATA_KEY: metadata, "training": {"registry_name": PINNED}},
        [FakeArtifact(native_bundle)],
    )
    resolver = RegistryResolver(api=FakeApi(runs={"team/project/student": run}))
    with pytest.raises(ValueError, match="REGISTRY explicitly"):
        resolver.for_run("wandb://team/project/student")
    assert run.used_artifacts_calls == 0


@pytest.mark.parametrize("case", ["none", "ambiguous", "mutable", "malformed"])
def test_legacy_recovery_rejects_missing_ambiguous_or_unpinned_artifacts(
    native_bundle, case
):
    artifacts = {
        "none": [],
        "ambiguous": [FakeArtifact(native_bundle), FakeArtifact(native_bundle, OTHER)],
        "mutable": [FakeArtifact(native_bundle, COLLECTION + ":latest")],
        "malformed": [FakeArtifact(native_bundle, "bad-reference")],
    }[case]
    run = FakeRun({"training": {"registry_name": COLLECTION + ":latest"}}, artifacts)
    resolver = RegistryResolver(api=FakeApi(runs={"team/project/legacy": run}))
    with pytest.raises(ValueError, match="REGISTRY explicitly"):
        resolver.for_run("wandb://team/project/legacy")


@pytest.mark.parametrize("training", [None, [], "obsolete-config"])
def test_malformed_legacy_training_config_reports_actionable_error(training):
    run = FakeRun({"training": training})
    resolver = RegistryResolver(api=FakeApi(runs={"team/project/legacy": run}))
    with pytest.raises(ValueError, match="REGISTRY explicitly"):
        resolver.for_run("wandb://team/project/legacy")


def test_explicit_override_never_reads_run_metadata(native_bundle):
    checkpoints, results = prepare_inputs(
        "eval_student", "wandb://team/project/student", native_bundle.as_uri()
    )
    assert checkpoints == ["wandb://team/project/student"]
    assert results[0].reference == native_bundle.as_uri()


@pytest.mark.parametrize("mode", ["distill", "eval_teacher", "eval_student"])
def test_auto_with_local_checkpoint_fails_before_network(tmp_path, mode):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    with pytest.raises(ValueError, match="REGISTRY=auto requires explicit wandb://"):
        prepare_inputs(mode, str(checkpoint), "auto")


def test_multiteacher_explicit_count_mismatch_fails_before_network():
    with pytest.raises(ValueError, match="one REGISTRY per TEACHER_CHECKPOINT"):
        prepare_inputs(
            "distill", "wandb://team/project/one,wandb://team/project/two", PINNED
        )


def test_rank_zero_records_unique_remote_usage_and_full_ordered_metadata(
    tmp_path, isolated_services
):
    local = ResolvedRegistry((tmp_path / "local").as_uri(), tmp_path / "local")
    first = ResolvedRegistry(PINNED, tmp_path / "cache-one")
    second = ResolvedRegistry(OTHER, tmp_path / "cache-two")
    inputs = [second, local, first, second]
    run = FakeRun(
        {"training": {"seed": 7, "registry_name": "obsolete"}, "unrelated": "keep"}
    )
    isolated_services.run = run

    record_registry_usage(inputs, "wandb")

    names = [r.reference for r in inputs]
    assert run.use_calls == [(OTHER, ARTIFACT_TYPE), (PINNED, ARTIFACT_TYPE)]
    assert run.config[METADATA_KEY] == {"version": 1, "registry_names": names}
    assert run.config["training"] == {"seed": 7, "registry_name": ",".join(names)}
    assert run.config["unrelated"] == "keep"
    assert run.config.allow_val_change is True
    assert "cache-one" not in str(run.config)


@pytest.mark.parametrize(
    "logger,rank",
    [("disabled", "0"), ("tensorboard", "0"), ("wandb", "1"), ("wandb", "7")],
)
def test_non_wandb_and_nonzero_ranks_have_no_wandb_side_effects(
    tmp_path, monkeypatch, logger, rank
):
    # An absent module is stricter than an API mock: these paths need no W&B installation.
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setenv("RANK", rank)
    record_registry_usage([ResolvedRegistry(PINNED, tmp_path)], logger)


def test_local_usage_records_metadata_without_remote_links(
    native_bundle, isolated_services
):
    run = FakeRun()
    isolated_services.run = run
    inputs = RegistryResolver().resolve(native_bundle.as_uri())
    record_registry_usage(inputs, "wandb")
    assert run.use_calls == []
    assert run.config[METADATA_KEY]["registry_names"] == [native_bundle.as_uri()]


def test_recording_without_active_wandb_run_is_a_noop(tmp_path):
    record_registry_usage([ResolvedRegistry(PINNED, tmp_path)], "wandb")


def test_training_hook_keeps_registry_identity_in_saved_checkpoint(
    native_bundle, tmp_path, isolated_services
):
    import torch
    from holosoma.config_types.logger import WandbLoggerConfig
    from holosoma.utils.eval_utils import CheckpointConfig

    from wbt_training.config_values.experiment import g1_wbt_terrain_ref_noscale
    from wbt_training.utils.checkpoint import (
        apply_preprocess_hook,
        load_saved_experiment_config,
    )

    alias = COLLECTION + ":latest"
    api = FakeApi({alias: FakeArtifact(native_bundle)})
    isolated_services.Api = lambda: api
    training_run = FakeRun({"training": {"registry_name": alias, "seed": 42}})
    isolated_services.run = training_run
    config = dataclasses.replace(
        g1_wbt_terrain_ref_noscale,
        logger=WandbLoggerConfig(),
        training=dataclasses.replace(
            g1_wbt_terrain_ref_noscale.training,
            registry_name=alias,
            num_envs=1,
            preprocess_hook_kwargs=json.dumps(
                {"add_onpath_obstacle": True, "num_variants": 1, "obstacle_seed": 7}
            ),
        ),
    )
    processed = apply_preprocess_hook(config)
    assert processed.training.registry_name == PINNED
    assert config.training.registry_name == alias
    assert training_run.use_calls == [(PINNED, ARTIFACT_TYPE)]
    assert training_run.config[METADATA_KEY]["registry_names"] == [PINNED]
    checkpoint = tmp_path / "model_1.pt"
    torch.save({"experiment_config": processed.to_serializable_dict()}, checkpoint)
    restored, _ = load_saved_experiment_config(
        CheckpointConfig(checkpoint=str(checkpoint))
    )
    assert restored.training.registry_name == PINNED
    assert api.artifact_calls == [alias]
    Path(processed.terrain.terrain_term.obj_file_path).unlink()


def test_generated_php_motion_upload_download_convert_and_recover(
    tmp_path, monkeypatch, isolated_services
):
    """Use real packaged G1 FK and upload APIs; only the W&B transport is fake."""
    pytest.importorskip(
        "mujoco",
        reason="Generated-motion integration needs motion-matching dependencies",
    )
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "motion_matching"))
    tracking = importlib.import_module("motion_matching.utils.tracking_format")
    uploader = importlib.import_module("motion_matching.upload_dataset")
    frames = 5
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 0] = 1
    qpos[:, 4] = np.linspace(0, 0.12, frames)
    qpos[:, 6] = 0.75
    qpos[:, 7:] = np.arange(29)[None, :] * np.arange(frames)[:, None] * 0.001
    generated = tracking.qpos_to_tracking(qpos, 50)
    generated["vel_cmd"] = (
        np.arange(frames * 15, dtype=np.float32).reshape(frames, 15) / 10
    )
    source = tmp_path / "generated"
    source.mkdir()
    np.savez(source / "sample_motion.npz", **generated)
    terrain = write_terrain(source / "sample_terrain.npy")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()

    class UploadArtifact(FakeArtifact):
        def __init__(self, *, name, type, metadata):
            super().__init__(uploaded, f"team/parkour_registry/{name}:v2", kind=type)
            self.metadata = metadata

        def add_file(self, path, *, name):
            destination = uploaded / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)

        def wait(self):
            return self

    class UploadRun:
        def __init__(self):
            self.links = []
            self.finishes = []

        def log_artifact(self, artifact):
            self.logged = artifact
            return artifact

        def link_artifact(self, *, artifact, target_path):
            self.links.append(target_path)
            self.linked = FakeArtifact(uploaded, PINNED, version=artifact.version)
            return self.linked

        def finish(self, *, exit_code):
            self.finishes.append(exit_code)

    upload_run = UploadRun()
    isolated_services.setup = lambda: SimpleNamespace(
        settings=SimpleNamespace(project=None)
    )
    isolated_services.init = lambda **kwargs: upload_run
    isolated_services.Artifact = UploadArtifact
    dataset = uploader.prepare_dataset(source)
    reference = uploader.upload_dataset(
        dataset, registry_name="stairs", organization="public-org"
    )
    assert reference == PINNED
    assert upload_run.links == [COLLECTION]
    assert upload_run.finishes == [0]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert {p.name: p.read_bytes() for p in uploaded.iterdir()} == before

    api = FakeApi({reference: upload_run.linked})
    resolver = RegistryResolver(api=api)
    _, registries = prepare_inputs("teacher", "", reference, resolver=resolver)
    counts, motion_path, terrain_path, _ = (
        multi_motion_helpers.pull_paired_from_wandb_registry(
            [r.local_uri for r in registries]
        )
    )
    assert counts == [1]
    with np.load(motion_path) as native:
        np.testing.assert_allclose(native["joint_pos"][:, :3], qpos[:, 4:7], atol=1e-6)
        np.testing.assert_allclose(native["joint_pos"][:, 3:7], qpos[:, :4], atol=1e-6)
        np.testing.assert_allclose(native["joint_pos"][:, 7:], qpos[:, 7:], atol=1e-6)
        np.testing.assert_allclose(
            native["joint_vel"][:, 6:],
            np.gradient(qpos[:, 7:], 1 / 50, axis=0),
            atol=1e-6,
        )
        np.testing.assert_array_equal(native["vel_cmd"], generated["vel_cmd"])
    with np.load(terrain_path) as combined:
        np.testing.assert_array_equal(combined["obj_list"], terrain)
        np.testing.assert_array_equal(combined["obj_count_list"], [0, 1])

    training_run = FakeRun()
    isolated_services.run = training_run
    record_registry_usage(registries, "wandb")
    assert training_run.use_calls == [(PINNED, ARTIFACT_TYPE)]
    api.runs["team/project/trained"] = training_run
    shutil.rmtree(registries[0].directory)
    _, recovered = prepare_inputs(
        "eval_student", "wandb://team/project/trained", "auto", resolver=resolver
    )
    assert [r.reference for r in recovered] == [PINNED]
    assert len(upload_run.linked.download_calls) == 2
    with np.load(recovered[0].directory / "sample_motion.npz") as native:
        np.testing.assert_array_equal(native["vel_cmd"], generated["vel_cmd"])
