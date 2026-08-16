"""Create ComfyUI ConvRot W4A4 Safetensors from a high-precision source."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch


PROFILE_PATTERNS = {
    "gemma": re.compile(
        r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    "qwen": re.compile(
        r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    # HunyuanVideo 1.5 double-stream blocks. Names follow the checkpoint's own convention;
    # HunyuanVideo.process_unet_state_dict rewrites them to the ComfyUI module names after
    # convert_old_quants has injected the .comfy_quant keys, and its substring replacements
    # ("_attn_qkv." -> "_attn.qkv.", "mlp.fc1." -> "mlp.0.", ...) carry the injected
    # .comfy_quant and .weight_scale keys along with the weights.
    "hunyuan_video_15": re.compile(
        r"^double_blocks\.\d+\.(?:(?:img|txt)_attn_(?:qkv|proj)|(?:img|txt)_mlp\.fc[12])\.weight$"
    ),
}
EXCLUSIONS = {
    "gemma": ["embed_tokens", "norm", "lm_head", "vision"],
    "qwen": ["embed_tokens", "norm", "lm_head", "visual", "vision"],
    # adaLN modulation drives every block's conditioning, so *_mod.linear stays high precision
    # along with the norms, the embedders, the token refiner, and the byt5/vision/time adapters.
    "hunyuan_video_15": [
        "_mod.linear", "_norm", "norm", "img_in", "txt_in", "byt5_in", "vision_in",
        "time_in", "final_layer", "_embedding", "task_bias",
    ],
}
HIGH_PRECISION_DTYPES = {"BF16", "F16", "F32"}
TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}
CONVROT_GROUP_SIZE = 256
QUANT_GROUP_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["auto", *PROFILE_PATTERNS], default="auto")
    parser.add_argument("--auto-detect", action="store_true", help="Alias for --profile auto")
    parser.add_argument("--dry-run", action="store_true", help="Plan the conversion without writing tensors")
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


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError


def normal_comfy_backend(comfy_root: Path) -> dict:
    code = f"""
import json, sys, torch, comfy_kitchen as ck
sys.path.insert(0, {str(comfy_root)!r})
import comfy.quant_ops
w = torch.empty((64, 64), device='cuda', dtype=torch.float16)
q = torch.empty((64, 32), device='cuda', dtype=torch.int8)
s = torch.empty((64,), device='cuda', dtype=torch.float32)
x = torch.empty((2, 64), device='cuda', dtype=torch.float16)
q_impl = ck.registry.get_implementation('quantize_convrot_w4a4_weight', kwargs={{'weight': w, 'convrot_groupsize': 64, 'quant_group_size': 64, 'stochastic_rounding': 0}})
l_impl = ck.registry.get_implementation('convrot_w4a4_linear', kwargs={{'x': x, 'qweight': q, 'wscales': s, 'bias': None, 'convrot_groupsize': 64, 'quant_group_size': 64, 'linear_dtype': 'int4'}})
print(json.dumps({{'quantizer': q_impl.__module__, 'linear': l_impl.__module__, 'backends': ck.list_backends()}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=comfy_root.parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["warning"] = result.stderr.strip()
    payload["native_ready"] = all(".backends.cuda" in payload[key] for key in ("quantizer", "linear"))
    return payload


def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        raw_header = json.loads(handle.read(header_size))
    metadata = dict(raw_header.pop("__metadata__", {}) or {})
    header = {
        name: {
            "dtype": info["dtype"],
            "shape": list(info["shape"]),
            "data_offsets": list(info["data_offsets"]),
        }
        for name, info in raw_header.items()
    }
    return header, metadata


def detect_profile(path: Path, names: list[str]) -> str:
    name_set = set(names)
    # Structural first: this mirrors the HunyuanVideo branch of comfy.model_detection, so the
    # profile is derived from the checkpoint rather than from its file name.
    if {
        "txt_in.individual_token_refiner.blocks.0.norm1.weight",
        "double_blocks.0.img_attn_qkv.weight",
    } <= name_set:
        return "hunyuan_video_15"
    lowered = path.name.lower()
    if "gemma" in lowered and any(name.startswith("model.layers.") for name in names):
        return "gemma"
    if "qwen" in lowered and any(name.startswith("model.layers.") for name in names):
        return "qwen"
    raise ValueError("auto-detection found no supported profile; pass a supported --profile after verifying the architecture")


def selected_layers(header: dict, profile: str) -> list[str]:
    pattern = PROFILE_PATTERNS[profile]
    selected = []
    for name, info in header.items():
        shape = info["shape"]
        if (
            pattern.fullmatch(name)
            and info["dtype"] in HIGH_PRECISION_DTYPES
            and len(shape) == 2
            and shape[1] % CONVROT_GROUP_SIZE == 0
        ):
            selected.append(name)
    return selected


def tensor_elements(shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def estimate_output(header: dict, source_size: int, selected: list[str]) -> int:
    dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2}
    estimate = source_size
    for name in selected:
        info = header[name]
        elements = tensor_elements(info["shape"])
        source_bytes = elements * dtype_bytes[info["dtype"]]
        packed_bytes = elements // 2
        scale_bytes = info["shape"][0] * 4
        estimate += packed_bytes + scale_bytes - source_bytes
    return estimate


def output_header(header: dict, metadata: dict[str, str], selected: list[str], layers: dict) -> tuple[dict, int]:
    selected_set = set(selected)
    quant_metadata = {"format_version": "1.0", "layers": layers}
    output_metadata = dict(metadata)
    output_metadata["_quantization_metadata"] = json.dumps(quant_metadata, separators=(",", ":"))
    output_metadata["quantization"] = "ConvRot W4A4"

    result = {"__metadata__": output_metadata}
    offset = 0
    for name, info in header.items():
        if name in selected_set:
            rows, columns = info["shape"]
            packed_bytes = rows * columns // 2
            result[name] = {
                "dtype": "I8",
                "shape": [rows, columns // 2],
                "data_offsets": [offset, offset + packed_bytes],
            }
            offset += packed_bytes
            scale_name = f"{name.removesuffix('.weight')}.weight_scale"
            scale_bytes = rows * 4
            result[scale_name] = {
                "dtype": "F32",
                "shape": [rows],
                "data_offsets": [offset, offset + scale_bytes],
            }
            offset += scale_bytes
        else:
            tensor_bytes = info["data_offsets"][1] - info["data_offsets"][0]
            result[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [offset, offset + tensor_bytes],
            }
            offset += tensor_bytes
    return result, offset


def encoded_header(header: dict) -> bytes:
    payload = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return payload + b" " * (-len(payload) % 8)


def copy_range(source_handle, output_handle, start: int, size: int) -> None:
    source_handle.seek(start)
    remaining = size
    while remaining:
        chunk = source_handle.read(min(16 * 1024**2, remaining))
        if not chunk:
            raise EOFError(f"Unexpected end of source with {remaining} bytes remaining")
        output_handle.write(chunk)
        remaining -= len(chunk)


def read_tensor_range(source_handle, start: int, size: int, dtype: str, shape: list[int]) -> tuple[torch.Tensor, bytearray]:
    source_handle.seek(start)
    raw = bytearray(size)
    view = memoryview(raw)
    position = 0
    while position < size:
        count = source_handle.readinto(view[position:])
        if not count:
            raise EOFError(f"Unexpected end of source with {size - position} bytes remaining")
        position += count
    tensor = torch.frombuffer(raw, dtype=TORCH_DTYPES[dtype]).reshape(shape)
    return tensor, raw


def write_tensor(output_handle, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    output_handle.write(memoryview(array).cast("B"))


def write_streamed_checkpoint(
    source: Path,
    partial: Path,
    header: dict,
    metadata: dict[str, str],
    selected: list[str],
    layers: dict,
    ck,
) -> None:
    target_header, expected_data_bytes = output_header(header, metadata, selected, layers)
    header_bytes = encoded_header(target_header)
    selected_set = set(selected)
    with source.open("rb") as source_handle:
        source_header_size = struct.unpack("<Q", source_handle.read(8))[0]
        source_data_start = 8 + source_header_size
        with partial.open("xb") as output_handle:
            output_handle.write(struct.pack("<Q", len(header_bytes)))
            output_handle.write(header_bytes)
            data_start = output_handle.tell()
            for index, (name, info) in enumerate(header.items(), 1):
                if name in selected_set:
                    start, end = info["data_offsets"]
                    source_tensor, raw = read_tensor_range(
                        source_handle,
                        source_data_start + start,
                        end - start,
                        info["dtype"],
                        info["shape"],
                    )
                    weight = source_tensor.to(device="cuda")
                    qdata, scales = ck.quantize_convrot_w4a4_weight(
                        weight,
                        convrot_groupsize=CONVROT_GROUP_SIZE,
                        quant_group_size=QUANT_GROUP_SIZE,
                        stochastic_rounding=0,
                    )
                    qdata = qdata.cpu().contiguous()
                    scales = scales.cpu().contiguous()
                    expected_q_shape = (info["shape"][0], info["shape"][1] // 2)
                    if qdata.dtype != torch.int8 or tuple(qdata.shape) != expected_q_shape:
                        raise RuntimeError(f"Unexpected packed weight for {name}: {qdata.dtype} {tuple(qdata.shape)}")
                    if scales.dtype != torch.float32 or tuple(scales.shape) != (info["shape"][0],):
                        raise RuntimeError(f"Unexpected scales for {name}: {scales.dtype} {tuple(scales.shape)}")
                    write_tensor(output_handle, qdata)
                    write_tensor(output_handle, scales)
                    del source_tensor, raw, weight, qdata, scales
                    torch.cuda.empty_cache()
                    print(f"[{index}/{len(header)}] quantized {name}", flush=True)
                else:
                    start, end = info["data_offsets"]
                    copy_range(source_handle, output_handle, source_data_start + start, end - start)
            actual_data_bytes = output_handle.tell() - data_start
            if actual_data_bytes != expected_data_bytes:
                raise RuntimeError(
                    f"Output data length mismatch: wrote {actual_data_bytes}, expected {expected_data_bytes}"
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())


def version_info(comfy_root: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(comfy_root), "rev-parse", "--short", "HEAD"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.strip()
    return {
        "comfy_version": commit or "unknown",
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    comfy_root = (args.comfy_root or default_comfy_root())
    if comfy_root is None or not (comfy_root / "comfy" / "quant_ops.py").is_file():
        raise SystemExit("Could not locate a ComfyUI checkout; pass --comfy-root")
    comfy_root = comfy_root.resolve()
    output = (args.output or source.with_name(f"{source.stem}_w4a4_convrot.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    if output == source:
        raise SystemExit("Refusing to overwrite the source model")
    if output.exists() or sidecar.exists():
        raise SystemExit(f"Refusing to overwrite existing output or sidecar: {output}")

    backend = None
    if not args.dry_run:
        backend = normal_comfy_backend(comfy_root)
        if not backend["native_ready"]:
            raise SystemExit(
                "Refusing conversion: normal ComfyUI resolves ConvRot to a non-CUDA backend. "
                f"quantizer={backend['quantizer']}, linear={backend['linear']}"
            )

    header, metadata = read_header(source)
    if metadata.get("_quantization_metadata"):
        raise SystemExit("Refusing to requantize a checkpoint that already has quantization metadata")
    profile = detect_profile(source, list(header)) if args.profile == "auto" else args.profile
    selected = selected_layers(header, profile)
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")
    estimated_size = estimate_output(header, source.stat().st_size, selected)
    print(f"Source: {source}")
    print(f"Profile: {profile}")
    print(f"Selected Linear weights: {len(selected)}")
    print(f"Estimated output: {human_size(estimated_size)}")
    print(f"Output: {output}")
    if args.dry_run:
        for name in selected[:20]:
            print(f"  {name}: {header[name]['shape']} {header[name]['dtype']}")
        if len(selected) > 20:
            print(f"  ... {len(selected) - 20} more")
        return 0


    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    free_disk = shutil.disk_usage(output.parent).free
    free_ram = psutil.virtual_memory().available
    if free_disk < estimated_size + 1024**3:
        raise SystemExit(f"Insufficient disk: {human_size(free_disk)} free, {human_size(estimated_size)} estimated")
    largest_selected = max(
        header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
        for name in selected
    )
    required_ram = largest_selected * 3 + 2 * 1024**3
    if free_ram < required_ram:
        raise SystemExit(
            f"Insufficient available RAM for streaming conversion: {human_size(free_ram)} free, "
            f"{human_size(required_ram)} required"
        )

    import comfy_kitchen as ck

    started = time.perf_counter()
    layers = {
        name.removesuffix(".weight"): {
            "format": "convrot_w4a4",
            "convrot_groupsize": CONVROT_GROUP_SIZE,
        }
        for name in selected
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise SystemExit(f"Refusing to overwrite stale partial output: {partial}")
    try:
        write_streamed_checkpoint(source, partial, header, metadata, selected, layers, ck)
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()

    elapsed = time.perf_counter() - started
    versions = version_info(comfy_root)
    manifest = {
        "source": str(source),
        "source_size": source.stat().st_size,
        "output": str(output),
        "output_size": output.stat().st_size,
        "architecture": profile,
        "quantization": "ConvRot W4A4",
        "layout": "TensorCoreConvRotW4A4Layout",
        "backend": backend["linear"],
        "expected_kernel": "native INT4 MMA",
        "weight_storage_dtype": "INT8 packed signed INT4",
        "activation_input_dtype": "BF16/FP16; dynamically rotated and quantized to INT4 in kernel",
        "quantized_tensors": len(selected),
        "preserved_tensors": len(header) - len(selected),
        "preserved_dtype": "original BF16/FP16/FP32",
        "excluded_patterns": EXCLUSIONS[profile],
        **versions,
        "conversion_seconds": round(elapsed, 3),
    }
    sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({human_size(output.stat().st_size)})")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
