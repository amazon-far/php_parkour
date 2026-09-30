"""Release copies preserve model execution and remove only release provenance."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from wbt_training.utils.release_metadata import (
    PROVENANCE_KEYS,
    main,
    sanitize_file,
    sanitize_model,
)


def _model():
    """A conditional model with weights, subgraphs and a local function."""
    branches = [
        helper.make_graph(
            [helper.make_node(op, ["x", "weight"], ["y"])],
            op,
            [],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        )
        for op in ("Add", "Sub")
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["result"],
                then_branch=branches[0],
                else_branch=branches[1],
            )
        ],
        "conditional",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 3])],
        [
            numpy_helper.from_array(
                np.arange(6, dtype=np.float32).reshape(2, 3), name="weight"
            )
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9
    )
    model.functions.append(
        helper.make_function(
            "release.test",
            "IdentityFn",
            ["X"],
            ["Y"],
            [helper.make_node("Identity", ["X"], ["Y"])],
            [helper.make_opsetid("", 18)],
        )
    )
    helper.set_model_props(
        model, {"robot_urdf": "<robot name='test'/>", "kp": "1,2,3", "iteration": "42"}
    )
    return model


def _annotate(model):
    model.doc_string = "model export trace"
    model.graph.doc_string = "graph export trace"
    model.graph.node[0].doc_string = "node export trace"
    model.graph.input[0].doc_string = "input export trace"
    model.graph.initializer[0].doc_string = "tensor export trace"
    for attribute in model.graph.node[0].attribute:
        attribute.g.doc_string = "subgraph export trace"
        attribute.g.node[0].doc_string = "nested node export trace"
        if "metadata_props" in attribute.g.DESCRIPTOR.fields_by_name:
            attribute.g.metadata_props.add(key="run_path", value="training/run")
    model.functions[0].doc_string = "function export trace"
    model.functions[0].node[0].doc_string = "function node export trace"
    for key in PROVENANCE_KEYS:
        model.metadata_props.add(key=key, value="training provenance")
    # Also remove duplicates; do not collapse or reorder other runtime entries.
    model.metadata_props.add(key="run_path", value="second run")


def _session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def test_recursive_metadata_and_exact_structure():
    expected = _model()
    original = _model()
    _annotate(original)
    before = original.SerializeToString()
    clean = sanitize_model(original)
    assert clean == expected
    assert original.SerializeToString() == before
    assert sanitize_model(clean) == clean
    onnx.checker.check_model(clean)
    for condition in (True, False):
        feeds = {
            "x": np.full((2, 3), 0.125, dtype=np.float32),
            "condition": np.array(condition),
        }
        np.testing.assert_array_equal(
            _session(original).run(None, feeds)[0], _session(clean).run(None, feeds)[0]
        )


def test_cli_recursive_copies_and_source_immutability(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "nested").mkdir()
    annotated = _model()
    _annotate(annotated)
    for relative in ("first.onnx", "nested/second.onnx"):
        onnx.save(annotated, source / relative)
    (source / "training.pt").write_bytes(b"not a release artifact")
    originals = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "release"
    assert main([str(source), "--output-dir", str(output)]) == 0
    assert sorted(str(p.relative_to(output)) for p in output.rglob("*.onnx")) == [
        "first.onnx",
        "nested/second.onnx",
    ]
    assert not (output / "training.pt").exists()
    for path in output.rglob("*.onnx"):
        assert onnx.load(path) == _model()
    assert all(path.read_bytes() == payload for path, payload in originals.items())


def test_cli_refuses_overwrites_collisions_and_nested_output(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    model_path = source / "model.onnx"
    onnx.save(_model(), model_path)
    original = model_path.read_bytes()
    for args in (
        [str(model_path)],  # an explicit output mode is required
        [str(source), "--output-dir", str(source / "release")],
        [str(model_path), "--output-dir", str(source)],
        [str(model_path), str(model_path), "--output-dir", str(tmp_path / "release")],
        [str(model_path), "--output-dir", str(tmp_path / "release"), "--inplace"],
    ):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code != 0
    assert model_path.read_bytes() == original
    assert not (tmp_path / "release").exists()
    assert not (source / "release").exists()


def test_explicit_inplace_and_file_api_guards(tmp_path):
    path = tmp_path / "model.onnx"
    original = _model()
    _annotate(original)
    onnx.save(original, path)
    path.chmod(0o640)
    with pytest.raises(FileExistsError):
        sanitize_file(path, path)
    with pytest.raises(ValueError, match="identical"):
        sanitize_file(path, tmp_path / "other.onnx", inplace=True)
    assert main([str(path), "--inplace"]) == 0
    assert onnx.load(path) == _model()
    assert path.stat().st_mode & 0o777 == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx"]


def test_external_weights_embedded_without_modifying_sources(tmp_path):
    source = tmp_path / "external.onnx"
    onnx.save_model(
        _model(),
        source,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    destination = tmp_path / "release" / "model.onnx"
    sanitize_file(source, destination)
    clean = onnx.load(destination, load_external_data=False)
    assert clean.graph.initializer[0].data_location == TensorProto.DEFAULT
    assert not clean.graph.initializer[0].external_data
    assert clean == onnx.load(
        source
    )  # re-embedding explicitly sets data_location=DEFAULT
    assert all(path.read_bytes() == payload for path, payload in before.items())
    onnx.checker.check_model(clean)
