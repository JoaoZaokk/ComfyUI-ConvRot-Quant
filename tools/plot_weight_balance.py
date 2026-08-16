"""Plot how two checkpoints' weight distributions diverge, and what int4 does to each.

Four panels, all computed from weights alone:

  1. Fraction of a layer's total weight *energy* carried beyond x times its RMS. Energy, not
     count: one weight at 40 rms is a rounding error by count and the whole problem by energy,
     and it is energy the per-row scale has to reach. This panel is the cause of panel 3.
  2. Crest factor (max|w| / rms) per sampled layer.
  3. Occupancy of the 15 int4 codes under W4A4's exact scheme (ConvRot, one absmax scale per
     row), with per-group-16 drawn for reference. A flat line is a fully used grid.
  4. The tools/weight_balance.py error ladder, as grouped bars.

    python tools/plot_weight_balance.py \
        --model "HunyuanVideo 1.5=path/hv15.safetensors" \
        --model "Gemma 3 12B=path/gemma.safetensors" \
        --out docs/weight_divergence.png
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_w4a8 import (  # noqa: E402
    HIGH_PRECISION_DTYPES,
    PROFILE_PATTERNS,
    detect_profile,
    read_header,
    read_tensor,
)
from weight_balance import INT4_MAX, LEVELS, rel_l2, uniform_int4  # noqa: E402

MAG_EDGES = np.linspace(0.0, 32.0, 321)  # |w| / rms
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help='"Label=path/to.safetensors", repeatable')
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def collect(label: str, path: Path, args: argparse.Namespace) -> dict:
    import comfy_kitchen as ck
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard, _rotate_weight

    header, _ = read_header(path)
    profile = detect_profile(path, list(header))
    pattern = PROFILE_PATTERNS[profile]
    names = [
        n for n, i in header.items()
        if pattern.fullmatch(n) and i["dtype"] in HIGH_PRECISION_DTYPES and len(i["shape"]) == 2
        and i["shape"][1] % args.convrot_groupsize == 0 and i["shape"][1] % args.group_size == 0
    ]
    if args.layers < len(names):
        step = len(names) / args.layers
        names = [names[int(i * step)] for i in range(args.layers)]

    energy = np.zeros(len(MAG_EDGES) - 1)
    codes_w4a4 = np.zeros(LEVELS)
    codes_group = np.zeros(LEVELS)
    crests, ladder = [], {"A": [], "B": [], "C": [], "D": []}

    print(f"{label}: {profile}, {len(names)} layers", flush=True)
    with path.open("rb") as handle:
        data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
        for name in names:
            info = header[name]
            start, end = info["data_offsets"]
            w = read_tensor(handle, data_start + start, end - start,
                            info["dtype"], info["shape"]).to(args.device, torch.float32)
            h = _build_hadamard(args.convrot_groupsize, device=w.device, dtype=torch.float32)
            rotated = _rotate_weight(w, h, args.convrot_groupsize)

            rms = w.pow(2).mean().sqrt()
            crests.append((w.abs().max() / rms).item())
            # Energy per bin, so the tail is weighted by w^2 rather than by how many weights
            # happen to be out there. bincount takes weights; histc does not.
            edges = torch.as_tensor(MAG_EDGES[1:-1], device=w.device, dtype=torch.float32)
            flat = w.flatten()
            index = torch.bucketize(flat.abs() / rms, edges)
            energy += torch.bincount(index, weights=flat.pow(2),
                                     minlength=len(MAG_EDGES) - 1).cpu().numpy() / flat.pow(2).sum().item()

            cb, rec_b = uniform_int4(rotated, None)
            cc, rec_c = uniform_int4(rotated, args.group_size)
            for codes, bucket in ((cb, codes_w4a4), (cc, codes_group)):
                hist = torch.bincount((codes.flatten().long() + INT4_MAX), minlength=LEVELS)
                bucket += hist.cpu().numpy() / codes.numel()

            _, hat_a = uniform_int4(w, None)
            ladder["A"].append(rel_l2(w, hat_a))
            ladder["B"].append(rel_l2(w, _rotate_weight(rec_b, h, args.convrot_groupsize)))
            ladder["C"].append(rel_l2(w, _rotate_weight(rec_c, h, args.convrot_groupsize)))
            qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
                w.to(torch.bfloat16), group_size=args.group_size,
                convrot_groupsize=args.convrot_groupsize, symmetric=True,
                scale_dtype=torch.float8_e4m3fn, codebook=True, codebook_tensor=None,
                stochastic_rounding=0)
            ladder["D"].append(rel_l2(w, ck.dequantize_w4a8_int8_weight(
                qdata, s_rel, s_channel, codebook, correction, group_size=args.group_size,
                convrot_groupsize=args.convrot_groupsize, output_dtype=torch.float32)))
            del w, rotated
            torch.cuda.empty_cache()

    n = len(names)
    return {
        "label": label,
        "energy": energy / n,
        "codes_w4a4": codes_w4a4 / n,
        "codes_group": codes_group / n,
        "crests": crests,
        "ladder": {k: float(np.mean(v)) for k, v in ladder.items()},
    }


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; pass --device cpu")

    models = []
    for spec in args.model:
        label, _, raw = spec.partition("=")
        path = Path(raw).resolve()
        if not path.is_file():
            raise SystemExit(f"No such file: {path}")
        models.append(collect(label, path, args))

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    fig.suptitle("Weight divergence and what int4 does to it", fontsize=14, y=0.985)
    centers = (MAG_EDGES[:-1] + MAG_EDGES[1:]) / 2

    ax = axes[0][0]
    for model, color in zip(models, COLORS):
        # Survival: share of the layer's total w^2 living beyond this many RMS.
        survival = model["energy"][::-1].cumsum()[::-1]
        visible = survival > 1e-9
        ax.semilogy(centers[visible], survival[visible] * 100, color=color, lw=1.6,
                    label=model["label"])
        crest = float(np.median(model["crests"]))
        ax.axvline(crest, color=color, ls="--", lw=1, alpha=0.6)
        ax.annotate(f"median crest {crest:.0f}\nthe W4A4 scale\nreaches to here",
                    xy=(crest + 0.5, 2e-5), fontsize=7.5, color=color, va="top")
    ax.set_title("1 · Share of a layer's weight energy out in the tail",
                 loc="left", fontweight="bold")
    ax.set_xlabel("distance from zero, in layer RMS")
    ax.set_ylabel("% of total $\\sum w^2$ beyond this point")
    ax.set_xlim(0, 32);  ax.set_ylim(1e-7, 200)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[0][1]
    for model, color in zip(models, COLORS):
        ax.plot(range(1, len(model["crests"]) + 1), model["crests"], "o-", color=color,
                lw=1.4, ms=4, label=model["label"])
    ax.set_title("2 · Crest factor per layer  ·  max|w| / rms", loc="left", fontweight="bold")
    ax.set_xlabel("sampled layer, input to output");  ax.set_ylabel("crest factor")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(4.0, color="#888", ls=":", lw=1)
    ax.annotate("Gaussian row of this width would sit near 4",
                xy=(1, 4.6), fontsize=8, color="#555")

    ax = axes[1][0]
    codes = np.arange(-INT4_MAX, INT4_MAX + 1)
    for model, color in zip(models, COLORS):
        ax.plot(codes, model["codes_w4a4"] * 100, "o-", color=color, lw=1.5, ms=4,
                label=f"{model['label']} · W4A4")
    ax.plot(codes, models[0]["codes_group"] * 100, "s--", color="#059669", lw=1.4, ms=3,
            label="per-group-16 (W4A8), for reference")
    ax.axhline(100 / LEVELS, color="#888", ls=":", lw=1)
    ax.annotate("a fully used 15-level grid", xy=(-6.8, 100 / LEVELS + 1.2), fontsize=8, color="#555")
    ax.set_title("3 · Where the weights land on the int4 grid", loc="left", fontweight="bold")
    ax.set_xlabel("int4 code");  ax.set_ylabel("% of weights")
    ax.set_xticks(codes);  ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    rungs = [("A\nper-row\nno rotation", "A"), ("B\nper-row\n+ConvRot\n= W4A4", "B"),
             ("C\nper-group-16\n+ConvRot", "C"), ("D\n+Lloyd-Max\n= W4A8", "D")]
    width = 0.8 / len(models)
    x = np.arange(len(rungs))
    for index, (model, color) in enumerate(zip(models, COLORS)):
        values = [model["ladder"][key] for _, key in rungs]
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width * 0.92, color=color, label=model["label"])
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_title("4 · Weight error after a quantize → dequantize round trip",
                 loc="left", fontweight="bold")
    ax.set_ylabel("relative L2 vs the source weight")
    ax.set_xticks(x, [label for label, _ in rungs], fontsize=8)
    ax.set_ylim(0, max(m["ladder"]["A"] for m in models) * 1.28)
    ax.legend(frameon=False, fontsize=8)

    fig.text(0.5, 0.005,
             "Weights only — no inference. Rung B is cross-checked against comfy-kitchen's own "
             "quantize/dequantize_convrot_w4a4_weight.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.018, 1, 0.975))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    for model in models:
        print(f"  {model['label']}: crest median "
              f"{np.median(model['crests']):.1f}, ladder "
              + "  ".join(f"{k}={v:.4f}" for k, v in model["ladder"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
