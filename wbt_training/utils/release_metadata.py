"""Copy ONNX models for release with runtime metadata intact and provenance removed.

Usage::

    python -m wbt_training.utils.release_metadata model.onnx --output-dir release
    python -m wbt_training.utils.release_metadata exports/ --output-dir release

Directory inputs are searched recursively; their relative paths are retained in
output-dir. File inputs use their basenames. Collisions and existing outputs are
errors. Only the separate, explicit --inplace mode replaces source files.
"""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import onnx
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

PROVENANCE_KEYS = frozenset(
    {"robot_urdf_path", "experiment_config", "wandb_run_path", "run_path"}
)


def _strip_metadata(message: Message) -> None:
    # Traverse the protobuf, including graph-valued attributes, local functions,
    # training graphs, tensors and value-info. Keep all non-provenance fields.
    for field, value in message.ListFields():
        if field.name == "doc_string":
            message.ClearField(field.name)
        elif field.name == "metadata_props":
            retained = [prop for prop in value if prop.key not in PROVENANCE_KEYS]
            del value[:]
            value.extend(retained)
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            if isinstance(value, Message):
                _strip_metadata(value)
            else:
                for child in value:
                    _strip_metadata(child)


def sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a sanitized copy, leaving the caller's model and tensors intact."""
    result = onnx.ModelProto()
    result.CopyFrom(model)
    _strip_metadata(result)
    return result


def sanitize_file(
    source: Path | str, destination: Path | str, *, inplace: bool = False
) -> Path:
    """Write a self-contained sanitized ONNX; never overwrite without inplace.

    External tensor data, if present, is loaded and embedded into the output.
    Source sidecar files are never modified. Inplace writes replace the ONNX
    atomically, preserving its permission bits.
    """
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    if inplace:
        if source != destination:
            raise ValueError(
                "Inplace mode requires the source and destination to be identical"
            )
    elif source == destination or destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite {destination}; use a new output directory"
        )

    model = onnx.load(source, load_external_data=True)
    onnx.external_data_helper.convert_model_from_external_data(model)
    payload = sanitize_model(model).SerializeToString()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if inplace:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
        try:
            temporary.write_bytes(payload)
            temporary.chmod(stat.S_IMODE(source.stat().st_mode))
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        # Exclusive creation also protects a concurrent writer and symlink targets.
        with destination.open("xb") as f:
            f.write(payload)
    return destination


def _plan_outputs(
    inputs: Iterable[Path], output_dir: Path | None
) -> list[tuple[Path, Path]]:
    jobs = []
    destinations = set()
    sources = set()
    for raw in inputs:
        source = raw.expanduser().resolve(strict=True)
        if source.is_dir():
            if output_dir is not None and (
                source == output_dir or source in output_dir.parents
            ):
                raise ValueError(
                    "Output directory must be outside each input directory"
                )
            files = sorted(
                p
                for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() == ".onnx"
            )
            if not files:
                raise ValueError(f"No ONNX models found in {source}")
            pairs = [(p, p.relative_to(source)) for p in files]
        elif source.is_file() and source.suffix.lower() == ".onnx":
            pairs = [(source, Path(source.name))]
        else:
            raise ValueError(f"Expected an ONNX file or directory: {source}")
        for candidate, relative in pairs:
            file = candidate.resolve(strict=True)
            destination = (
                file if output_dir is None else (output_dir / relative).resolve()
            )
            if destination in destinations or file in sources:
                raise ValueError(f"Duplicate input or output path: {destination}")
            if output_dir is not None:
                if output_dir not in destination.parents:
                    raise ValueError(
                        f"Output path escapes the output directory: {destination}"
                    )
                if destination.exists() or destination == file:
                    raise FileExistsError(
                        f"Refusing to overwrite {destination}; use a new output directory"
                    )
            jobs.append((file, destination))
            sources.add(file)
            destinations.add(destination)
    if output_dir is not None and sources & destinations:
        raise ValueError("Output files must not replace any input files")
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="ONNX files or directories (recursive)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit directory for new copies; existing files are refused",
    )
    mode.add_argument(
        "--inplace",
        action="store_true",
        help="Explicitly replace the input ONNX files atomically",
    )
    args = parser.parse_args(argv)
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else None
    )
    try:
        jobs = _plan_outputs(args.inputs, output_dir)
        for source, destination in jobs:
            sanitize_file(source, destination, inplace=args.inplace)
            print(f"{source} -> {destination}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}{os.linesep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
