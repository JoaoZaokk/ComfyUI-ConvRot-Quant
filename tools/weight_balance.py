"""Measure how unbalanced a checkpoint's Linear weights are, and what that costs in int4.

Two questions, both answered arithmetically from the weights alone -- no GPU inference, no
prompts, no images:

1. *How unbalanced is the distribution?*  Excess kurtosis, crest factor, tail energy, and how
   much of a per-row int4 grid an average row actually occupies (Shannon entropy of the code
   histogram, reported as effective bits out of the 3.907 that 15 levels can carry).

2. *What does that imbalance cost?*  An ablation ladder over the exact same weight, each rung
   adding one defence, measured as relative Frobenius error of a quantize -> dequantize round
   trip:

       A  uniform int4, one scale per row, no rotation      naive baseline
       B  uniform int4, one scale per row, + ConvRot        == what convrot_w4a4 does
       C  uniform int4, per-group-16 scale, + ConvRot       isolates group size
       D  Lloyd-Max int4, per-group-16 + ALS, + ConvRot     == what asym_w4a8_int8 does

   B is computed twice -- once here, once through comfy_kitchen's own
   ``quantize/dequantize_convrot_w4a4_weight`` -- and the two must agree, which is what makes
   A, C and D trustworthy. D likewise runs through comfy_kitchen's real W4A8 path.

The rotation and codebook internals are read from ``comfy_kitchen.backends.eager`` rather than
reimplemented, so this measures the library that actually ships, not an idealisation of it.

    python tools/weight_balance.py --input model.safetensors
    python tools/weight_balance.py --input model.safetensors --layers 32 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_w4a8 import (  # same directory
    HIGH_PRECISION_DTYPES,
    PROFILE_PATTERNS,
    detect_profile,
    read_header,
    read_tensor,
)

INT4_MAX = 7
LEVELS = 2 * INT4_MAX + 1  # -7..7, what quantize_signed_int4_rowwise clamps to
MAX_BITS = math.log2(LEVELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--profile", choices=["auto", *PROFILE_PATTERNS], default="auto")
    parser.add_argument("--layers", type=int, default=12,
                        help="how many layers to sample, spread evenly (0 = all)")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


# --- distribution shape -------------------------------------------------------------------

def excess_kurtosis(x: torch.Tensor) -> float:
    """E[((x-mu)/sigma)^4] - 3.  Zero for a Gaussian, positive for heavy tails.

    Same estimator comfy_kitchen's codebook gate uses, so the numbers are comparable to its
    -0.1 threshold.
    """
    x = x.flatten().float()
    return (((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0).item()


def shape_stats(x: torch.Tensor) -> dict:
    x = x.float()
    flat = x.flatten()
    sigma = flat.std()
    rms = flat.pow(2).mean().sqrt()
    absmax = flat.abs().max()
    tail = flat[flat.abs() > 4 * sigma]
    # Energy, not count: one weight at 40 sigma is a rounding error by count and the whole
    # problem by energy, and it is energy the scale has to cover.
    tail_energy = (tail.pow(2).sum() / flat.pow(2).sum()).item() if tail.numel() else 0.0
    row_absmax = x.abs().amax(dim=-1).clamp(min=1e-10)
    row_rms = x.pow(2).mean(dim=-1).sqrt()
    return {
        "kurtosis": excess_kurtosis(flat),
        "crest": (absmax / rms).item(),
        "tail_count_frac": tail.numel() / flat.numel(),
        "tail_energy_frac": tail_energy,
        "row_rms_over_absmax": (row_rms / row_absmax).mean().item(),
    }


def effective_bits(codes: torch.Tensor) -> float:
    """Shannon entropy of the int4 code histogram, in bits.

    You always pay 4 bits per weight. This is what you get back. A row whose scale is set by a
    lone outlier crushes every other weight toward code 0, and the entropy collapses.
    """
    hist = torch.bincount((codes.flatten().long() + INT4_MAX), minlength=LEVELS).float()
    p = hist / hist.sum()
    p = p[p > 0]
    return (-(p * p.log2()).sum()).item()


# --- the ablation ladder ------------------------------------------------------------------

def rel_l2(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    reference = reference.float()
    return ((approximation.float() - reference).norm() / reference.norm()).item()


def uniform_int4(x: torch.Tensor, group_size: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric uniform int4 with an absmax scale per row (group_size None) or per group.

    The per-row branch is quantize_signed_int4_rowwise's arithmetic, restated so the ladder's
    rungs differ in exactly one variable.
    """
    rows, cols = x.shape
    grouped = x.reshape(rows, cols // group_size, group_size) if group_size else x
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / INT4_MAX).clamp(min=1e-10)
    codes = (grouped / scale).round().clamp(-INT4_MAX, INT4_MAX)
    return codes.reshape(rows, cols), (codes * scale).reshape(rows, cols)


def measure_layer(weight: torch.Tensor, args: argparse.Namespace, hadamard, rotate) -> dict:
    import comfy_kitchen as ck

    w = weight.to(args.device, torch.float32)
    h = hadamard(args.convrot_groupsize, device=w.device, dtype=torch.float32)
    rotated = rotate(w, h, args.convrot_groupsize)

    # What the codebook gate sees: rotated, then divided by its per-group absmax.
    grouped = rotated.view(w.shape[0], w.shape[1] // args.group_size, args.group_size)
    normalized = grouped / grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    codes_a, hat_a = uniform_int4(w, None)
    codes_b, rec_b = uniform_int4(rotated, None)
    codes_c, rec_c = uniform_int4(rotated, args.group_size)

    result = {
        "shape": list(w.shape),
        "raw": shape_stats(w),
        "rotated": shape_stats(rotated),
        "kurtosis_group_normalized": excess_kurtosis(normalized),
        "bits_A_row_norot": effective_bits(codes_a),
        "bits_B_row_convrot": effective_bits(codes_b),
        "bits_C_group16_convrot": effective_bits(codes_c),
        "relL2_A_row_norot": rel_l2(w, hat_a),
        "relL2_B_row_convrot": rel_l2(w, rotate(rec_b, h, args.convrot_groupsize)),
        "relL2_C_group16_convrot": rel_l2(w, rotate(rec_c, h, args.convrot_groupsize)),
    }

    # B, through comfy_kitchen itself. Must match the hand-computed rung.
    qdata, scales = ck.quantize_convrot_w4a4_weight(
        w.to(torch.bfloat16), convrot_groupsize=args.convrot_groupsize, quant_group_size=64)
    result["relL2_B_library"] = rel_l2(w, ck.dequantize_convrot_w4a4_weight(
        qdata, scales, convrot_groupsize=args.convrot_groupsize, quant_group_size=64,
        output_dtype=torch.float32))

    # D, the real W4A8 weight path.
    qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
        w.to(torch.bfloat16), group_size=args.group_size,
        convrot_groupsize=args.convrot_groupsize, symmetric=True,
        scale_dtype=torch.float8_e4m3fn, codebook=True, codebook_tensor=None,
        stochastic_rounding=0)
    result["relL2_D_w4a8"] = rel_l2(w, ck.dequantize_w4a8_int8_weight(
        qdata, s_rel, s_channel, codebook, correction, group_size=args.group_size,
        convrot_groupsize=args.convrot_groupsize, output_dtype=torch.float32))
    result["codebook_fitted"] = (
        codebook is not None
        and abs(result["kurtosis_group_normalized"]) > 0  # recorded for the reader
        and result["kurtosis_group_normalized"] > -0.1
    )
    return result


def mean_of(rows: list[dict], key: str) -> float:
    parts = key.split(".")
    values = []
    for row in rows:
        value = row
        for part in parts:
            value = value[part]
        values.append(value)
    return sum(values) / len(values)


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"No such file: {source}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; pass --device cpu (slow)")

    header, _ = read_header(source)
    profile = detect_profile(source, list(header)) if args.profile == "auto" else args.profile
    pattern = PROFILE_PATTERNS[profile]
    names = [
        name for name, info in header.items()
        if pattern.fullmatch(name)
        and info["dtype"] in HIGH_PRECISION_DTYPES
        and len(info["shape"]) == 2
        and info["shape"][1] % args.convrot_groupsize == 0
        and info["shape"][1] % args.group_size == 0
    ]
    if not names:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")
    if args.layers and args.layers < len(names):
        step = len(names) / args.layers
        names = [names[int(i * step)] for i in range(args.layers)]

    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard, _rotate_weight

    print(f"{source.name}   profile={profile}   sampling {len(names)} layers")
    print(f"group_size={args.group_size}  convrot_groupsize={args.convrot_groupsize}\n")
    head = (f"{'layer':<46}{'kurt':>7}{'crest':>7}{'tail%':>7}"
            f"{'bitsB':>7}{'bitsC':>7}{'A':>8}{'B':>8}{'C':>8}{'D':>8}")
    print(head)
    print("-" * len(head))

    rows = []
    with source.open("rb") as handle:
        data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
        for name in names:
            info = header[name]
            start, end = info["data_offsets"]
            weight = read_tensor(handle, data_start + start, end - start,
                                 info["dtype"], info["shape"])
            row = measure_layer(weight, args, _build_hadamard, _rotate_weight)
            row["name"] = name
            rows.append(row)
            drift = abs(row["relL2_B_library"] - row["relL2_B_row_convrot"])
            flag = "" if drift < 5e-3 else f"  !! B drift {drift:.4f}"
            print(f"{name[:45]:<46}{row['raw']['kurtosis']:>7.2f}"
                  f"{row['raw']['crest']:>7.1f}"
                  f"{row['raw']['tail_energy_frac'] * 100:>7.2f}"
                  f"{row['bits_B_row_convrot']:>7.2f}{row['bits_C_group16_convrot']:>7.2f}"
                  f"{row['relL2_A_row_norot']:>8.4f}{row['relL2_B_row_convrot']:>8.4f}"
                  f"{row['relL2_C_group16_convrot']:>8.4f}{row['relL2_D_w4a8']:>8.4f}{flag}",
                  flush=True)
            del weight
            torch.cuda.empty_cache()

    print(f"\n{'':<46}{'mean':>7}")
    print(f"  raw excess kurtosis                         {mean_of(rows, 'raw.kurtosis'):>7.2f}")
    print(f"  after ConvRot rotation                      {mean_of(rows, 'rotated.kurtosis'):>7.2f}")
    print(f"  after rotation + per-group-16 normalization {mean_of(rows, 'kurtosis_group_normalized'):>7.2f}"
          "   (gate fires above -0.10)")
    print(f"  crest factor  max|w| / rms                  {mean_of(rows, 'raw.crest'):>7.1f}")
    print(f"  energy beyond 4 sigma                       {mean_of(rows, 'raw.tail_energy_frac') * 100:>7.2f} %")
    print(f"  rms / absmax per row                        {mean_of(rows, 'raw.row_rms_over_absmax'):>7.3f}")
    print(f"\n  effective bits (of {MAX_BITS:.3f} paid)")
    for label, key in (("A  per-row, no rotation      ", "bits_A_row_norot"),
                       ("B  per-row, ConvRot   [W4A4] ", "bits_B_row_convrot"),
                       ("C  per-group-16, ConvRot     ", "bits_C_group16_convrot")):
        print(f"    {label}{mean_of(rows, key):>7.3f}")
    print("\n  relative L2 of the weight round trip")
    baseline = mean_of(rows, "relL2_A_row_norot")
    for label, key in (("A  uniform, per-row, no rotation        ", "relL2_A_row_norot"),
                       ("B  uniform, per-row, ConvRot     [W4A4] ", "relL2_B_row_convrot"),
                       ("C  uniform, per-group-16, ConvRot       ", "relL2_C_group16_convrot"),
                       ("D  Lloyd-Max, per-group-16, ConvRot [W4A8]", "relL2_D_w4a8")):
        value = mean_of(rows, key)
        print(f"    {label:<42}{value:>8.4f}   {baseline / value:>5.2f}x vs A")
    drift = max(abs(r["relL2_B_library"] - r["relL2_B_row_convrot"]) for r in rows)
    print(f"\n  max |B here - B via comfy_kitchen| = {drift:.6f}"
          f"   {'agrees' if drift < 5e-3 else 'DISAGREES - ladder is not trustworthy'}")

    if args.json:
        args.json.write_text(json.dumps(
            {"source": str(source), "profile": profile, "group_size": args.group_size,
             "convrot_groupsize": args.convrot_groupsize, "layers": rows},
            indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
