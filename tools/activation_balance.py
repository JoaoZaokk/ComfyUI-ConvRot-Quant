"""Measure what int4 does to real activations, captured from a live forward pass.

tools/weight_balance.py answered the weight half: ConvRot W4A4 recovers ~2.9 of the 4 bits it
pays, because one absmax scale per row must cover a crest factor of 12-28. This answers the
activation half, which is the one nobody measures because it needs a forward pass.

ConvRot's activation path is `quantize_signed_int4_rowwise(rotate(x))` -- **one scale per token**,
spanning every channel, 15 uniform levels. A single outlier channel therefore sets the scale for
the whole token vector. This hooks the real kernel, captures the real tensors it was handed, and
reports:

  * crest factor per token, and the per-channel outlier ratio that causes it
  * Shannon entropy of the int4 codes, as effective bits out of the 3.907 paid
  * round-trip error under the exact shipped scheme, against three alternatives:
    per-group-16 int4 (finer scale), rowwise int8 (what W4A8 runs), per-group-16 int8

Run it against a convrot_w4a4 checkpoint; the hook only fires when that kernel does.

    python tools/activation_balance.py --text-encoder gemma_w4a4_convrot.safetensors \\
        --prompt "List the first 8 prime numbers." --samples 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

INT4_MAX = 7
INT4_LEVELS = 2 * INT4_MAX + 1
INT4_BITS = math.log2(INT4_LEVELS)
INT8_MAX = 127
INT8_BITS = math.log2(2 * INT8_MAX + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--prompt", default="List the first 8 prime numbers, then explain in one "
                                            "sentence what makes a number prime.")
    parser.add_argument("--samples", type=int, default=24, help="activation tensors to keep")
    parser.add_argument("--max-tokens", type=int, default=12, help="tokens generated while capturing")
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def entropy_bits(codes: torch.Tensor, levels: int, offset: int) -> float:
    hist = torch.bincount((codes.flatten().long() + offset), minlength=levels).float()
    p = hist / hist.sum()
    p = p[p > 0]
    return (-(p * p.log2()).sum()).item()


def quantize_rowwise(x: torch.Tensor, limit: int, group_size: int | None):
    """Symmetric absmax quantization; group_size None reproduces the shipped per-row scale."""
    rows, cols = x.shape
    grouped = x.reshape(rows, cols // group_size, group_size) if group_size else x
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / limit).clamp(min=1e-10)
    codes = (grouped / scale).round().clamp(-limit, limit)
    return codes.reshape(rows, cols), (codes * scale).reshape(rows, cols)


def rel_l2(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    return ((approximation - reference).norm() / reference.norm().clamp(min=1e-12)).item()


def measure(x: torch.Tensor, args: argparse.Namespace, hadamard, rotate) -> dict:
    from comfy_kitchen.backends.eager.convrot_w4a4 import _rotate_activation  # noqa: F401

    x = x.reshape(-1, x.shape[-1]).float()
    h = hadamard(args.convrot_groupsize, device=x.device, dtype=torch.float32)
    rotated = rotate(x, h, args.convrot_groupsize)

    row_absmax = x.abs().amax(dim=-1).clamp(min=1e-10)
    row_rms = x.pow(2).mean(dim=-1).sqrt()
    # Persistent-channel outliers are the SmoothQuant signature: one column hot across all tokens.
    channel_absmax = x.abs().amax(dim=0)
    channel_ratio = (channel_absmax.max() / channel_absmax.median().clamp(min=1e-10)).item()

    codes_a4, rec_a4 = quantize_rowwise(rotated, INT4_MAX, None)
    codes_a4g, rec_a4g = quantize_rowwise(rotated, INT4_MAX, args.group_size)
    codes_a8, rec_a8 = quantize_rowwise(rotated, INT8_MAX, None)
    _, rec_a8g = quantize_rowwise(rotated, INT8_MAX, args.group_size)

    return {
        "rows": x.shape[0],
        "channels": x.shape[1],
        "crest_row_mean": (row_absmax / row_rms.clamp(min=1e-10)).mean().item(),
        "channel_outlier_ratio": channel_ratio,
        "bits_int4_rowwise": entropy_bits(codes_a4, INT4_LEVELS, INT4_MAX),
        "bits_int4_group": entropy_bits(codes_a4g, INT4_LEVELS, INT4_MAX),
        "bits_int8_rowwise": entropy_bits(codes_a8, 2 * INT8_MAX + 1, INT8_MAX),
        "zero_code_fraction": (codes_a4 == 0).float().mean().item(),
        "relL2_int4_rowwise": rel_l2(rotated, rec_a4),
        "relL2_int4_group": rel_l2(rotated, rec_a4g),
        "relL2_int8_rowwise": rel_l2(rotated, rec_a8),
        "relL2_int8_group": rel_l2(rotated, rec_a8g),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    import comfy_kitchen.tensor.convrot_w4a4 as convrot
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard, _rotate_activation

    captured: list[torch.Tensor] = []
    call_count = {"n": 0}
    original_linear = convrot.convrot_w4a4_linear

    def capturing(x, *rest, **kwargs):
        call_count["n"] += 1
        # Spread the sample across layers instead of taking the first N of layer 0.
        if len(captured) < args.samples and call_count["n"] % 7 == 1:
            captured.append(x.detach().reshape(-1, x.shape[-1])[:512].float().cpu())
        return original_linear(x, *rest, **kwargs)

    convrot.convrot_w4a4_linear = capturing

    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", args.text_encoder)
    clip = comfy.sd.load_clip(ckpt_paths=[path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.LTXV)
    formats = {getattr(m, "quant_format", None) for m in clip.cond_stage_model.modules()}
    formats.discard(None)
    if "convrot_w4a4" not in formats:
        raise SystemExit(f"Not a convrot_w4a4 checkpoint (found {formats or 'none'}); "
                         "the capture hook would never fire")

    tokens = clip.tokenize(args.prompt, skip_template=False, min_length=1)
    clip.generate(tokens, do_sample=False, max_length=args.max_tokens, temperature=1.0,
                  top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, seed=0)
    convrot.convrot_w4a4_linear = original_linear

    if not captured:
        raise SystemExit("No activations captured")
    print(f"\ncaptured {len(captured)} activation tensors from {call_count['n']} kernel calls\n")

    rows = [measure(x.cuda(), args, _build_hadamard, _rotate_activation) for x in captured]
    mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731

    print(f"  tokens x channels, per sample        {rows[0]['rows']} x {rows[0]['channels']}")
    print(f"  crest factor per token  max|x|/rms   {mean('crest_row_mean'):>8.1f}")
    print(f"  worst channel / median channel       {mean('channel_outlier_ratio'):>8.1f}")
    print()
    print(f"  effective bits, int4 per-token scale {mean('bits_int4_rowwise'):>8.3f}   "
          f"of {INT4_BITS:.3f} paid   <-- what W4A4 ships")
    print(f"  effective bits, int4 per-group-{args.group_size:<2}     {mean('bits_int4_group'):>8.3f}")
    print(f"  effective bits, int8 per-token scale {mean('bits_int8_rowwise'):>8.3f}   "
          f"of {INT8_BITS:.3f} paid   <-- what W4A8 ships")
    print(f"  share of activations landing on 0    {mean('zero_code_fraction') * 100:>8.1f} %")
    print()
    print("  relative L2 of the activation round trip")
    baseline = mean("relL2_int4_rowwise")
    for label, key in ((f"int4, one scale per token  [W4A4] ", "relL2_int4_rowwise"),
                       (f"int4, one scale per {args.group_size} channels ", "relL2_int4_group"),
                       (f"int8, one scale per token  [W4A8] ", "relL2_int8_rowwise"),
                       (f"int8, one scale per {args.group_size} channels ", "relL2_int8_group")):
        value = mean(key)
        print(f"    {label:<38}{value:>8.4f}   {baseline / max(value, 1e-9):>6.1f}x better than W4A4")

    if args.json:
        args.json.write_text(json.dumps(
            {"text_encoder": args.text_encoder, "prompt": args.prompt, "samples": rows},
            indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
