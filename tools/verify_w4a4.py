"""Verify a ComfyUI ConvRot W4A4 checkpoint and its optional source."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import torch


TORCH_DTYPES = {
    "I8": torch.int8,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--kernel-smoke", action="store_true")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        help="path to the ComfyUI checkout; auto-detected from this file's location if omitted",
    )
    return parser.parse_args()


def default_comfy_root() -> Path | None:
    """Find a ComfyUI checkout whether this lives in a portable root or in custom_nodes."""
    here = Path(__file__).resolve()
    candidates = []
    for parent in here.parents:
        candidates.append(parent)
        candidates.append(parent / "ComfyUI")
    for candidate in candidates:
        if (candidate / "comfy" / "quant_ops.py").is_file():
            return candidate
    return None


def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(header_size))
    metadata = dict(raw.pop("__metadata__", {}) or {})
    return raw, metadata


def load_tensor_cuda(path: Path, info: dict) -> torch.Tensor:
    start, end = info["data_offsets"]
    size = end - start
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_size + start)
        raw = bytearray(size)
        view = memoryview(raw)
        position = 0
        while position < size:
            count = handle.readinto(view[position:])
            if not count:
                raise EOFError(f"Unexpected end of {path} with {size - position} bytes remaining")
            position += count
    return torch.frombuffer(raw, dtype=TORCH_DTYPES[info["dtype"]]).reshape(info["shape"]).cuda()


def normal_comfy_backend(comfy_root: Path) -> dict:
    code = f"""
import json, sys, torch, comfy_kitchen as ck
sys.path.insert(0, {str(comfy_root)!r})
import comfy.quant_ops
q = torch.empty((64, 32), device='cuda', dtype=torch.int8)
s = torch.empty((64,), device='cuda', dtype=torch.float32)
x = torch.empty((2, 64), device='cuda', dtype=torch.float16)
impl = ck.registry.get_implementation('convrot_w4a4_linear', kwargs={{'x': x, 'qweight': q, 'wscales': s, 'bias': None, 'convrot_groupsize': 64, 'quant_group_size': 64, 'linear_dtype': 'int4'}})
print(json.dumps({{'linear': impl.__module__, 'backends': ck.list_backends()}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=comfy_root.parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["warning"] = result.stderr.strip()
    payload["native_ready"] = ".backends.cuda" in payload["linear"]
    return payload


def resolve_source(model: Path, explicit_source: Path | None) -> Path | None:
    if explicit_source:
        return explicit_source.resolve()
    sidecar = model.with_suffix(".quant.json")
    if sidecar.is_file():
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        source = Path(manifest.get("source", ""))
        return source if source.is_file() else None
    return None


def validate_structure(model_header: dict, metadata: dict, source_header: dict | None) -> list[str]:
    errors = []
    raw_quant = metadata.get("_quantization_metadata")
    if not raw_quant:
        return ["missing _quantization_metadata"]
    try:
        quant = json.loads(raw_quant)
    except json.JSONDecodeError as error:
        return [f"invalid _quantization_metadata JSON: {error}"]
    layers = quant.get("layers", {})
    if not layers:
        return ["quantization metadata contains no layers"]
    for layer_name, config in layers.items():
        if config.get("format") != "convrot_w4a4":
            errors.append(f"{layer_name}: unexpected format {config.get('format')!r}")
            continue
        weight_name = f"{layer_name}.weight"
        scale_name = f"{layer_name}.weight_scale"
        weight = model_header.get(weight_name)
        scale = model_header.get(scale_name)
        if not weight or not scale:
            errors.append(f"{layer_name}: missing packed weight or weight_scale")
            continue
        if weight.get("dtype") != "I8" or len(weight.get("shape", [])) != 2:
            errors.append(f"{layer_name}: packed weight must be 2D I8, got {weight}")
        if scale.get("dtype") not in {"F32", "F16", "BF16"} or len(scale.get("shape", [])) != 1:
            errors.append(f"{layer_name}: invalid scale {scale}")
        if source_header is not None:
            source_weight = source_header.get(weight_name)
            if not source_weight:
                errors.append(f"{layer_name}: weight missing from source")
            elif weight["shape"] != [source_weight["shape"][0], source_weight["shape"][1] // 2]:
                errors.append(f"{layer_name}: packed shape {weight['shape']} does not match source {source_weight['shape']}")
            if scale["shape"] != [weight["shape"][0]]:
                errors.append(f"{layer_name}: scale shape {scale['shape']} does not match rows {weight['shape'][0]}")
    if source_header is not None:
        quantized_weights = {f"{name}.weight" for name in layers}
        for name, source_info in source_header.items():
            if name in quantized_weights:
                continue
            output_info = model_header.get(name)
            if output_info is None or any(
                output_info.get(field) != source_info.get(field)
                for field in ("dtype", "shape")
            ):
                errors.append(f"{name}: preserved tensor dtype/shape changed")
    return errors


def data_start(path: Path) -> int:
    with path.open("rb") as handle:
        return 8 + struct.unpack("<Q", handle.read(8))[0]


def compare_ranges(left_handle, left_start: int, right_handle, right_start: int, size: int) -> bool:
    left_handle.seek(left_start)
    right_handle.seek(right_start)
    remaining = size
    while remaining:
        chunk_size = min(16 * 1024**2, remaining)
        if left_handle.read(chunk_size) != right_handle.read(chunk_size):
            return False
        remaining -= chunk_size
    return True


def validate_preserved_bytes(
    model: Path,
    source: Path,
    model_header: dict,
    source_header: dict,
    layers: dict,
) -> list[str]:
    errors = []
    quantized_weights = {f"{name}.weight" for name in layers}
    model_base = data_start(model)
    source_base = data_start(source)
    with model.open("rb") as model_handle, source.open("rb") as source_handle:
        for name, source_info in source_header.items():
            if name in quantized_weights:
                continue
            output_info = model_header[name]
            source_start, source_end = source_info["data_offsets"]
            output_start, output_end = output_info["data_offsets"]
            size = source_end - source_start
            if output_end - output_start != size or not compare_ranges(
                model_handle,
                model_base + output_start,
                source_handle,
                source_base + source_start,
                size,
            ):
                errors.append(f"{name}: preserved tensor bytes changed")
    return errors


def kernel_smoke(
    model: Path,
    source: Path,
    layer_name: str,
    model_header: dict,
    source_header: dict,
) -> dict:
    import comfy_kitchen as ck

    weight_name = f"{layer_name}.weight"
    scale_name = f"{layer_name}.weight_scale"
    qweight = load_tensor_cuda(model, model_header[weight_name])
    scale = load_tensor_cuda(model, model_header[scale_name])
    weight = load_tensor_cuda(source, source_header[weight_name])
    x = torch.randn((2, weight.shape[1]), device="cuda", dtype=weight.dtype)
    kwargs = {
        "x": x, "qweight": qweight, "wscales": scale, "bias": None,
        "convrot_groupsize": 256, "quant_group_size": 64, "linear_dtype": "int4",
    }
    implementation = ck.registry.get_implementation("convrot_w4a4_linear", kwargs=kwargs)
    output = ck.convrot_w4a4_linear(**kwargs)
    reference = torch.nn.functional.linear(x, weight)
    error = output.float() - reference.float()
    return {
        "layer": layer_name,
        "backend": f"{implementation.__module__}.{implementation.__name__}",
        "output_dtype": str(output.dtype),
        "relative_rmse": error.square().mean().sqrt().div(reference.float().square().mean().sqrt()).item(),
        "max_abs_error": error.abs().max().item(),
    }


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    comfy_root = (args.comfy_root or default_comfy_root())
    if comfy_root is None or not (comfy_root / "comfy" / "quant_ops.py").is_file():
        raise SystemExit("Could not locate a ComfyUI checkout; pass --comfy-root")
    comfy_root = comfy_root.resolve()
    source = resolve_source(model, args.source)
    model_header, metadata = read_header(model)
    source_header = read_header(source)[0] if source else None
    errors = validate_structure(model_header, metadata, source_header)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    quant = json.loads(metadata["_quantization_metadata"])
    layers = quant["layers"]
    if source:
        errors = validate_preserved_bytes(model, source, model_header, source_header, layers)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
    print(f"Structural verification: PASS ({len(layers)} ConvRot W4A4 layers)")
    print(f"Source comparison: {'PASS' if source else 'SKIPPED (no source supplied/found)'}")
    backend = normal_comfy_backend(comfy_root)
    print(f"Normal ComfyUI backend: {backend['linear']}")
    if not backend["native_ready"]:
        print("ERROR: normal ComfyUI is not selecting the CUDA ConvRot backend")
        return 1
    if args.kernel_smoke:
        if not source:
            print("ERROR: --kernel-smoke requires --source or a valid sidecar source")
            return 1
        result = kernel_smoke(model, source, next(iter(layers)), model_header, source_header)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
