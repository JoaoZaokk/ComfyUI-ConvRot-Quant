"""Convert a Gemma 3 checkpoint to ConvRot W4A4 with SmoothQuant folded into the norms.

The probe in tools/svdquant_probe.py says channel smoothing is the largest single term in the
SVDQuant recipe -- 1.80x on layer output, against 1.12x for the low-rank branch -- and that it
costs nothing at runtime because `lambda` folds into the RMSNorm that feeds the Linear. This
builds that checkpoint so the claim can be tested on an answer instead of on an error norm.

    Y = (x/lambda) @ (W*lambda).T          exact in full precision
    norm_weight  <- (norm_weight + 1)/lambda - 1      Gemma's RMSNorm is (1 + w), not w
    linear_weight <- linear_weight * lambda

Two details decide whether this is correct rather than merely plausible:

* **A norm has several consumers.** `input_layernorm` feeds q_proj, k_proj and v_proj;
  `pre_feedforward_layernorm` feeds gate_proj and up_proj. One lambda per norm, and every consumer
  must be compensated, so `weight_max` is taken across the whole group.
* **`o_proj` and `down_proj` have no norm directly ahead of them** -- they follow attention output
  and the gated product. They are left unsmoothed here rather than given a runtime multiply, which
  makes this exactly the free configuration: 5 of the 7 projections, no kernel change, no extra op.

Calibration runs the W4A4 checkpoint itself with its weights retyped so ComfyUI dequantizes them,
which gives full-precision activations off 4-bit weights at 7.6 GiB instead of loading the 21.9 GiB
source. Channel structure is a property of the model and survives that approximation; the shipped
`.quant.json` records that it was used.

    python tools/quant_w4a4_smooth.py --source gemma_..._it_heretic.safetensors \\
        --calibrate-with gemma_..._w4a4_convrot.safetensors --alpha 0.5
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from quant_w4a4 import human_size  # noqa: E402
from quant_w4a8 import SAFETENSORS_DTYPE, copy_range, read_header, read_tensor  # noqa: E402

# The two norm groups. Every consumer of a norm shares one lambda.
GROUPS = {
    "input_layernorm": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "pre_feedforward_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
}
# Quantized but never smoothed: no norm feeds them directly.
UNSMOOTHED = ("self_attn.o_proj", "mlp.down_proj")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

CALIBRATION_PROMPTS = [
    "List the first 8 prime numbers, then explain in one sentence what makes a number prime.",
    "Explain in two sentences why the sky appears blue.",
    "Write one short paragraph about a cat who learns to open doors.",
    "What is 47 times 89? Show the steps.",
    "Traduza para o português: the quick brown fox jumps over the lazy dog.",
    "Name three countries in South America and their capitals.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="the BF16 checkpoint")
    parser.add_argument("--calibrate-with", required=True,
                        help="a W4A4 file name inside models/text_encoders, used for calibration")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--calibration-tokens", type=int, default=24)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def calibrate(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Per-input-channel absmax of what each norm hands its consumers."""
    import comfy.sd
    import folder_paths
    from comfy.quant_ops import QuantizedTensor

    path = folder_paths.get_full_path_or_raise("text_encoders", args.calibrate_with)
    clip = comfy.sd.load_clip(ckpt_paths=[path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.LTXV)
    model = clip.cond_stage_model
    retyped = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if getattr(module, "quant_format", None) == "convrot_w4a4" and isinstance(weight, QuantizedTensor):
            module.weight = torch.nn.Parameter(weight.to(dtype=torch.float32), requires_grad=False)
            retyped += 1
    print(f"calibrating on {args.calibrate_with}, {retyped} weights forced to dequantize", flush=True)

    # One probe per group: q_proj and gate_proj see exactly their norm's output.
    probes = {"input_layernorm": "self_attn.q_proj", "pre_feedforward_layernorm": "mlp.gate_proj"}
    stats: dict[str, torch.Tensor] = {}
    handles = []
    for module_name, module in model.named_modules():
        match = re.search(r"model\.layers\.(\d+)\.(.+)$", module_name)
        if not match:
            continue
        for norm, probe in probes.items():
            if match.group(2) != probe:
                continue
            key = f"model.layers.{match.group(1)}.{norm}.weight"

            def hook(mod, inputs, key=key):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().abs().amax(dim=0)
                stats[key] = x.cpu() if key not in stats else torch.maximum(stats[key], x.cpu())
            handles.append(module.register_forward_pre_hook(hook))

    for index, prompt in enumerate(CALIBRATION_PROMPTS, 1):
        clip.generate(clip.tokenize(prompt, skip_template=False, min_length=1), do_sample=False,
                      max_length=args.calibration_tokens, temperature=1.0, top_k=0, top_p=1.0,
                      min_p=0.0, repetition_penalty=1.0, seed=0)
        print(f"  [{index}/{len(CALIBRATION_PROMPTS)}] {len(stats)} norms observed", flush=True)
    for handle in handles:
        handle.remove()
    del clip, model
    torch.cuda.empty_cache()
    return stats


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"No such source: {source}")
    output = (args.output or source.with_name(f"{source.stem}_w4a4_smooth.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    if output == source:
        raise SystemExit("Refusing to overwrite the source model")
    if output.exists() or sidecar.exists():
        raise SystemExit(f"Refusing to overwrite existing output or sidecar: {output}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    header, metadata = read_header(source)
    if metadata.get("_quantization_metadata"):
        raise SystemExit("Refusing to requantize a checkpoint that already carries metadata")

    layer_ids = sorted({int(LAYER_RE.match(k).group(1)) for k in header if LAYER_RE.match(k)})
    selected, norm_keys = [], []
    for layer in layer_ids:
        for norm, members in GROUPS.items():
            norm_keys.append(f"model.layers.{layer}.{norm}.weight")
            selected += [f"model.layers.{layer}.{m}.weight" for m in members]
        selected += [f"model.layers.{layer}.{m}.weight" for m in UNSMOOTHED]
    missing = [k for k in selected + norm_keys if k not in header]
    if missing:
        raise SystemExit(f"Source lacks {len(missing)} expected tensors, e.g. {missing[0]}")

    print(f"Source: {source}")
    print(f"Layers: {len(layer_ids)}   quantized: {len(selected)}   "
          f"smoothed: {len(selected) - 2 * len(layer_ids)}   alpha: {args.alpha}")
    print(f"Output: {output}")
    if args.dry_run:
        return 0

    stats = calibrate(args)
    if len(stats) != len(norm_keys):
        raise SystemExit(f"Calibration saw {len(stats)} norms, expected {len(norm_keys)}")

    import comfy_kitchen as ck

    started = time.perf_counter()
    quantized: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    new_norms: dict[str, torch.Tensor] = {}
    lambda_report = []

    def load(handle, data_start, key) -> torch.Tensor:
        info = header[key]
        start, end = info["data_offsets"]
        return read_tensor(handle, data_start + start, end - start, info["dtype"], info["shape"])

    def store(key, weight: torch.Tensor) -> None:
        qdata, wscales = ck.quantize_convrot_w4a4_weight(
            weight.to(device="cuda", dtype=torch.bfloat16),
            convrot_groupsize=args.convrot_groupsize, quant_group_size=64)
        quantized[key] = qdata.cpu().contiguous()
        scales[key] = wscales.cpu().contiguous()
        del qdata, wscales
        torch.cuda.empty_cache()

    with source.open("rb") as handle:
        data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
        for layer in layer_ids:
            for norm, members in GROUPS.items():
                norm_key = f"model.layers.{layer}.{norm}.weight"
                member_keys = [f"model.layers.{layer}.{m}.weight" for m in members]
                weights = {k: load(handle, data_start, k).cuda().float() for k in member_keys}
                act_max = stats[norm_key].cuda().clamp(min=1e-5)
                # One lambda per norm: weight_max spans every consumer of that norm.
                weight_max = torch.stack([w.abs().amax(dim=0) for w in weights.values()]).amax(0)
                lam = (act_max.pow(args.alpha) / weight_max.clamp(min=1e-5).pow(1 - args.alpha)
                       ).clamp(min=1e-5)
                for key, weight in weights.items():
                    store(key, weight * lam)
                norm_weight = load(handle, data_start, norm_key).cuda().float()
                # Gemma's RMSNorm applies (1 + w), so the fold has to go through that offset.
                new_norms[norm_key] = ((norm_weight + 1.0) / lam - 1.0).to(torch.bfloat16).cpu()
                lambda_report.append({
                    "norm": norm_key,
                    "act_channel_ratio_before": (act_max.max() / act_max.median()).item(),
                    "act_channel_ratio_after": ((act_max / lam).max() / (act_max / lam).median()).item(),
                })
                del weights, act_max, weight_max, lam, norm_weight
            for member in UNSMOOTHED:
                key = f"model.layers.{layer}.{member}.weight"
                store(key, load(handle, data_start, key).cuda().float())
            if (layer + 1) % 8 == 0 or layer == layer_ids[-1]:
                print(f"[{layer + 1}/{len(layer_ids)}] layers quantized", flush=True)

    before = sum(r["act_channel_ratio_before"] for r in lambda_report) / len(lambda_report)
    after = sum(r["act_channel_ratio_after"] for r in lambda_report) / len(lambda_report)
    print(f"activation channel outlier ratio, mean over {len(lambda_report)} norms: "
          f"{before:.0f} -> {after:.1f}")

    selected_set = set(selected)
    layers_meta = {k.removesuffix(".weight"): {"format": "convrot_w4a4",
                                               "convrot_groupsize": args.convrot_groupsize}
                   for k in selected}
    output_metadata = dict(metadata)
    output_metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": layers_meta}, separators=(",", ":"))
    output_metadata["quantization"] = "ConvRot W4A4"
    output_metadata["smoothquant_alpha"] = str(args.alpha)

    target = {"__metadata__": output_metadata}
    offset, plan = 0, []
    for name, info in header.items():
        if name in selected_set:
            for key, tensor in ((name, quantized[name]), (f"{name}_scale", scales[name])):
                nbytes = tensor.numel() * tensor.element_size()
                target[key] = {"dtype": SAFETENSORS_DTYPE[tensor.dtype], "shape": list(tensor.shape),
                               "data_offsets": [offset, offset + nbytes]}
                plan.append(("write", tensor))
                offset += nbytes
        elif name in new_norms:
            tensor = new_norms[name]
            nbytes = tensor.numel() * tensor.element_size()
            target[name] = {"dtype": "BF16", "shape": list(tensor.shape),
                            "data_offsets": [offset, offset + nbytes]}
            plan.append(("write", tensor.view(torch.int16)))
            offset += nbytes
        else:
            start, end = info["data_offsets"]
            size = end - start
            target[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [offset, offset + size]}
            plan.append(("copy", (start, size)))
            offset += size

    payload = json.dumps(target, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * (-len(payload) % 8)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise SystemExit(f"Refusing to overwrite stale partial: {partial}")
    try:
        with source.open("rb") as source_handle, partial.open("xb") as out_handle:
            data_start = 8 + struct.unpack("<Q", source_handle.read(8))[0]
            out_handle.write(struct.pack("<Q", len(payload)))
            out_handle.write(payload)
            body_start = out_handle.tell()
            for kind, item in plan:
                if kind == "write":
                    out_handle.write(memoryview(item.numpy()).cast("B"))
                else:
                    start, size = item
                    copy_range(source_handle, out_handle, data_start + start, size)
            if out_handle.tell() - body_start != offset:
                raise RuntimeError("length mismatch")
            out_handle.flush()
            os.fsync(out_handle.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()

    elapsed = time.perf_counter() - started
    sidecar.write_text(json.dumps({
        "source": str(source), "output": str(output), "output_size": output.stat().st_size,
        "quantization": "ConvRot W4A4 + SmoothQuant", "smoothquant_alpha": args.alpha,
        "calibrated_with": args.calibrate_with,
        "calibration_note": "activations captured from the W4A4 checkpoint with weights retyped "
                            "so ComfyUI dequantizes them: 4-bit weights, full-precision activations",
        "calibration_prompts": len(CALIBRATION_PROMPTS),
        "smoothed_projections": sorted({m for members in GROUPS.values() for m in members}),
        "quantized_unsmoothed": list(UNSMOOTHED),
        "quantized_tensors": len(selected),
        "act_channel_ratio_before": round(before, 2), "act_channel_ratio_after": round(after, 2),
        "convrot_groupsize": args.convrot_groupsize,
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "conversion_seconds": round(elapsed, 3),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({human_size(output.stat().st_size)}) in {elapsed:.1f} s")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
