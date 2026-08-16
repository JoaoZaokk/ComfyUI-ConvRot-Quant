"""Test the SVDQuant recipe arithmetically on real activations, before writing any kernel.

comfy-kitchen ships the SVDQuant *runtime* -- `quantize_svdquant_w4a4` and
`scaled_mm_svdquant_w4a4` -- but every input it needs from the weight side (`smooth`, `lora_down`,
`lora_up`, per-group `wscales`) has to be produced by a calibration pass that does not exist.
Building that is weeks of work. This answers whether it would pay, in one run, using arithmetic.

The metric is the only one that matters: relative L2 of the **layer output** `x @ W.T`, against
the bf16 source weight and the real activation the kernel was handed mid-generation. Activation
round-trip error is not the target -- SVDQuant deliberately *increases* weight error to decrease
activation error, so anything measured on one side alone can be gamed.

Ladder, each rung adding one piece of the recipe:

    0  bf16                                            reference (denominator)
    1  ConvRot W4A4, as shipped                        one absmax scale per token, 15 levels
    2  + SmoothQuant channel migration                 x/lambda, W*lambda -- moves the outlier
                                                       out of the activation and into the weight,
                                                       which is quantized offline where it is
                                                       affordable
    3  + rank-R low-rank branch in fp32                W*lambda = L1 L2 + R; L1 L2 runs in high
                                                       precision, only R goes to int4
    4  asym_w4a8_int8, as shipped                      what already works, for scale

Rungs 1 and 4 call comfy-kitchen's real quantizers and linears, so the hand-built rungs 2 and 3
are anchored to the shipping implementation rather than to a private reimplementation.

    python tools/svdquant_probe.py --text-encoder gemma_..._w4a4_convrot.safetensors \\
        --source gemma_..._it_heretic.safetensors --layers 8 --rank 32
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from quant_w4a8 import read_header, read_tensor  # noqa: E402

# transformer.model.layers.3.self_attn.q_proj -> model.layers.3.self_attn.q_proj.weight
MODULE_TO_KEY = re.compile(r"(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-encoder", required=True, help="a convrot_w4a4 file, for capture")
    parser.add_argument("--source", required=True, type=Path,
                        help="the high-precision checkpoint the W4A4 file was made from")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant exponent: lambda = max|x|^a / max|W|^(1-a)")
    parser.add_argument("--max-rows", type=int, default=256, help="tokens kept per layer")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--svdquant-group", type=int, default=64,
                        help="scale granularity along K for both operands, as SVDQuant uses")
    parser.add_argument("--alpha-sweep", type=float, nargs="*",
                        default=[0.3, 0.5, 0.65, 0.8, 0.9])
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def rel_l2(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    reference = reference.float()
    return ((approximation.float() - reference).norm() / reference.norm()).item()


def convrot_roundtrip(x: torch.Tensor, w: torch.Tensor, args) -> torch.Tensor:
    """One layer through comfy-kitchen's real ConvRot W4A4 quantizer and linear."""
    import comfy_kitchen as ck

    qweight, wscales = ck.quantize_convrot_w4a4_weight(
        w.to(torch.bfloat16), convrot_groupsize=args.convrot_groupsize, quant_group_size=64)
    return ck.convrot_w4a4_linear(x.to(torch.bfloat16), qweight, wscales, None,
                                  convrot_groupsize=args.convrot_groupsize,
                                  quant_group_size=64).float()


def w4a8_roundtrip(x: torch.Tensor, w: torch.Tensor, args) -> torch.Tensor:
    import comfy_kitchen as ck

    qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
        w.to(torch.bfloat16), group_size=args.group_size,
        convrot_groupsize=args.convrot_groupsize, symmetric=True,
        scale_dtype=torch.float8_e4m3fn, codebook=True, codebook_tensor=None,
        stochastic_rounding=0)
    return ck.w4a8_int8_linear(x.to(torch.bfloat16), qdata, s_rel, s_channel, codebook,
                               correction, None, group_size=args.group_size,
                               convrot_groupsize=args.convrot_groupsize,
                               out_dtype=torch.bfloat16).float()


def group_quant(t: torch.Tensor, group_size: int, limit: int = 7) -> torch.Tensor:
    """Symmetric int4 with one absmax scale per `group_size` values along K, then dequantized.

    This is the scale granularity SVDQuant uses on *both* operands -- `wscales` is (K//64, N) and
    `ascales` is (K//64, M) -- as opposed to ConvRot's single scale per weight row and per token.
    Dequantizing here is mathematically what the grouped int4 GEMM accumulates.
    """
    rows, cols = t.shape
    grouped = t.reshape(rows, cols // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / limit).clamp(min=1e-10)
    return ((grouped / scale).round().clamp(-limit, limit) * scale).reshape(rows, cols)


def grouped_linear(x: torch.Tensor, w: torch.Tensor, group_size: int) -> torch.Tensor:
    return group_quant(x, group_size) @ group_quant(w, group_size).T


def smooth_vector(x: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    """SmoothQuant per-input-channel migration factor, clamped away from zero."""
    act_max = x.abs().amax(dim=0).clamp(min=1e-5)
    weight_max = w.abs().amax(dim=0).clamp(min=1e-5)
    return (act_max.pow(alpha) / weight_max.pow(1.0 - alpha)).clamp(min=1e-5)


def measure_layer(name: str, x: torch.Tensor, w: torch.Tensor, args) -> dict:
    exact = (x.float() @ w.float().T)

    result = {"layer": name, "rows": x.shape[0], "shape": list(w.shape)}
    result["relL2_1_w4a4"] = rel_l2(exact, convrot_roundtrip(x, w, args))
    result["relL2_4_w4a8"] = rel_l2(exact, w4a8_roundtrip(x, w, args))

    # Rung 2: migrate the activation outlier into the weight. Exact in full precision --
    # (x / lambda) @ (w * lambda).T == x @ w.T -- so any change is entirely quantization.
    lam = smooth_vector(x, w, args.alpha)
    x_smooth = x.float() / lam
    w_smooth = w.float() * lam
    result["relL2_2_smooth"] = rel_l2(exact, convrot_roundtrip(x_smooth, w_smooth, args))
    result["act_channel_ratio_before"] = (
        x.abs().amax(dim=0).max() / x.abs().amax(dim=0).median().clamp(min=1e-9)).item()
    result["act_channel_ratio_after"] = (
        x_smooth.abs().amax(dim=0).max() / x_smooth.abs().amax(dim=0).median().clamp(min=1e-9)).item()

    # Rung 3: peel a rank-R piece off the smoothed weight and keep it in high precision, so
    # only the residual is asked to fit in int4.
    u, s, vh = torch.linalg.svd(w_smooth, full_matrices=False)
    rank = min(args.rank, s.numel())
    l1 = u[:, :rank] * s[:rank]
    l2 = vh[:rank]
    residual = w_smooth - l1 @ l2
    low_rank_output = (x_smooth @ l2.T) @ l1.T
    result["relL2_3_smooth_lowrank"] = rel_l2(
        exact, convrot_roundtrip(x_smooth, residual, args) + low_rank_output)
    result["energy_in_top_rank"] = (s[:rank].pow(2).sum() / s.pow(2).sum()).item()

    # Rungs 5-7: SVDQuant's other half. Its scales are per group of 64 along K on *both*
    # operands, which ConvRot does not do on either. Isolated here from the smoothing so the
    # two contributions can be told apart.
    gs = args.svdquant_group
    result["relL2_5_group_only"] = rel_l2(exact, grouped_linear(x.float(), w.float(), gs))
    result["relL2_6_group_smooth"] = rel_l2(exact, grouped_linear(x_smooth, w_smooth, gs))
    result["relL2_7_full"] = rel_l2(
        exact, grouped_linear(x_smooth, residual, gs) + low_rank_output)

    # alpha is the one free parameter in the recipe and 0.5 is only a convention. Sweeping it is
    # cheap here and decides whether smoothing alone can reach W4A8 without any low-rank branch.
    sweep = {}
    for alpha in args.alpha_sweep:
        lam_a = smooth_vector(x, w, alpha)
        xs, ws = x.float() / lam_a, w.float() * lam_a
        sweep[f"{alpha:.2f}"] = {
            "convrot_smooth": rel_l2(exact, convrot_roundtrip(xs, ws, args)),
            "group_smooth": rel_l2(exact, grouped_linear(xs, ws, gs)),
        }
    result["alpha_sweep"] = sweep
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not args.source.is_file():
        raise SystemExit(f"No such source: {args.source}")

    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", args.text_encoder)
    clip = comfy.sd.load_clip(ckpt_paths=[path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.LTXV)
    model = clip.cond_stage_model

    targets = {}
    for module_name, module in model.named_modules():
        match = MODULE_TO_KEY.search(module_name)
        if match and getattr(module, "quant_format", None) == "convrot_w4a4":
            targets[module] = f"{match.group(1)}.weight"
    if not targets:
        raise SystemExit("No convrot_w4a4 Linears found; is this a W4A4 checkpoint?")

    chosen = list(targets.items())
    step = max(1, len(chosen) // args.layers)
    chosen = dict(chosen[::step][:args.layers])
    print(f"capturing {len(chosen)} of {len(targets)} quantized Linears", flush=True)

    captured: dict[str, torch.Tensor] = {}
    handles = []
    for module, key in chosen.items():
        def hook(mod, inputs, key=key):
            if key not in captured:
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                captured[key] = x[: args.max_rows].float().cpu()
        handles.append(module.register_forward_pre_hook(hook))

    clip.generate(clip.tokenize("List the first 8 prime numbers.", skip_template=False, min_length=1),
                  do_sample=False, max_length=args.max_tokens, temperature=1.0, top_k=0,
                  top_p=1.0, min_p=0.0, repetition_penalty=1.0, seed=0)
    for handle in handles:
        handle.remove()
    del clip, model
    torch.cuda.empty_cache()

    header, _ = read_header(args.source)
    missing = [k for k in captured if k not in header]
    if missing:
        raise SystemExit(f"Source is missing {len(missing)} captured layers, e.g. {missing[0]}")

    rows = []
    head = f"{'layer':<42}{'W4A4':>9}{'+smooth':>9}{'+lowrank':>10}{'W4A8':>9}{'chan ratio':>22}"
    print(f"\n{head}\n{'-' * len(head)}")
    with args.source.open("rb") as handle:
        data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
        for key, x in captured.items():
            info = header[key]
            start, end = info["data_offsets"]
            w = read_tensor(handle, data_start + start, end - start,
                            info["dtype"], info["shape"]).cuda()
            row = measure_layer(key, x.cuda(), w, args)
            rows.append(row)
            print(f"{key.removeprefix('model.layers.')[:41]:<42}"
                  f"{row['relL2_1_w4a4']:>9.4f}{row['relL2_2_smooth']:>9.4f}"
                  f"{row['relL2_3_smooth_lowrank']:>10.4f}{row['relL2_4_w4a8']:>9.4f}"
                  f"{row['act_channel_ratio_before']:>11.0f} ->{row['act_channel_ratio_after']:>8.1f}",
                  flush=True)
            del w
            torch.cuda.empty_cache()

    mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
    baseline = mean("relL2_1_w4a4")
    print(f"\n  layer-output relative L2, mean of {len(rows)} layers "
          f"(rank {args.rank}, alpha {args.alpha})")
    for label, key in (("1  ConvRot W4A4, as shipped         ", "relL2_1_w4a4"),
                       ("2  + SmoothQuant migration          ", "relL2_2_smooth"),
                       (f"3  + rank-{args.rank} branch in fp32        ", "relL2_3_smooth_lowrank"),
                       ("4  asym_w4a8_int8, as shipped       ", "relL2_4_w4a8"),
                       (f"5  per-group-{args.svdquant_group} scales only, both sides", "relL2_5_group_only"),
                       (f"6  + SmoothQuant migration          ", "relL2_6_group_smooth"),
                       (f"7  + rank-{args.rank} branch  [full SVDQuant]", "relL2_7_full")):
        value = mean(key)
        print(f"    {label}{value:>8.4f}   {baseline / max(value, 1e-9):>6.2f}x better than rung 1")
    print(f"\n  activation channel outlier ratio   "
          f"{mean('act_channel_ratio_before'):>8.0f} -> {mean('act_channel_ratio_after'):.1f}")
    print(f"  weight energy inside the top {args.rank} singular values"
          f"   {mean('energy_in_top_rank') * 100:>6.2f} %")

    if args.json:
        args.json.write_text(json.dumps(
            {"source": str(args.source), "rank": args.rank, "alpha": args.alpha, "layers": rows},
            indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
