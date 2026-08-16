"""Chat with a Gemma 3 text encoder through ComfyUI's own loader and generation loop.

The point is not the chatbot. The weight-balance measurement showed Gemma's weights are *less*
balanced than a diffusion model's, yet the diffusion model is the one 4-bit destroys. This runs
the other half of that comparison: put the same 4-bit formats through the LLM and read the output.

Because the answer only means something if you know which kernel ran, every run counts native
ConvRot/W4A8 linear calls against weight dequantizations. A checkpoint that quietly dequantizes
is running W4-storage with full-precision activations -- a different experiment from W4A4.

    python tools/gemma_chat.py --text-encoder gemma_3_12B_it_heretic.safetensors \\
        --prompt "Explain what a Hadamard rotation does, in two sentences."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

QUANT_FORMATS = ("convrot_w4a4", "asym_w4a8_int8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-encoder", required=True,
                        help="file name inside models/text_encoders")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true",
                        help="argmax instead of sampling; makes two checkpoints directly comparable")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--force-native", action="store_true",
                        help="retype quantized weights to the execution dtype so the kernel "
                             "dispatches instead of dequantizing")
    parser.add_argument("--force-dequant", action="store_true",
                        help="the opposite: retype the quantized weights so they cannot match the "
                             "activation dtype, which makes comfy/ops.py dequantize them. Same "
                             "4-bit weights, full-precision activations -- i.e. W4A16, which is "
                             "what GPTQ/AWQ/GGUF actually ship. Isolates weight bits from "
                             "activation bits on one checkpoint.")
    parser.add_argument("--native-dtype", default="bfloat16",
                        choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def instrument() -> dict:
    """Count native quantized-linear calls against weight dequantizations, for both formats."""
    import comfy_kitchen.tensor.convrot_w4a4 as convrot
    import comfy_kitchen.tensor.w4a8_int8 as w4a8
    from comfy_kitchen.registry import registry

    counters = {"native_calls": 0, "dequant_calls": 0, "impls": set()}

    def wrap_linear(module, name: str, arg_names: tuple[str, ...]):
        original = getattr(module, name)

        def counting(*args, **kwargs):
            counters["native_calls"] += 1
            if len(counters["impls"]) < 4:
                probe = dict(zip(arg_names, args))
                probe.update(kwargs)
                try:
                    impl = registry.get_implementation(name, kwargs=probe)
                    counters["impls"].add(f"{impl.__module__}.{impl.__name__}")
                except Exception as error:  # probing must never break generation
                    counters["impls"].add(f"<probe failed: {type(error).__name__}>")
            return original(*args, **kwargs)

        setattr(module, name, counting)

    def wrap_dequant(layout):
        original = layout.dequantize.__func__

        def counting(cls, qdata, params):
            counters["dequant_calls"] += 1
            return original(cls, qdata, params)

        layout.dequantize = classmethod(counting)

    wrap_linear(convrot, "convrot_w4a4_linear", ("x", "qweight", "wscales", "bias"))
    wrap_linear(w4a8, "w4a8_int8_linear", ("x", "qweight", "s_rel", "s_channel"))
    wrap_dequant(convrot.TensorCoreConvRotW4A4Layout)
    wrap_dequant(w4a8.AsymW4A8Int8Layout)
    return counters


def quant_summary(model) -> dict:
    formats: dict[str, int] = {}
    for module in model.modules():
        quant_format = getattr(module, "quant_format", None)
        if quant_format is not None:
            formats[quant_format] = formats.get(quant_format, 0) + 1
    return formats


def retype_quantized(model, dtype) -> int:
    """QuantizedTensor.to(dtype=) only rewrites params.orig_dtype; the packed data is untouched."""
    from comfy.quant_ops import QuantizedTensor

    retyped = 0
    for module in model.modules():
        if getattr(module, "quant_format", None) not in QUANT_FORMATS:
            continue
        weight = module.weight
        if not isinstance(weight, QuantizedTensor) or weight.dtype == dtype:
            continue
        module.weight = torch.nn.Parameter(weight.to(dtype=dtype), requires_grad=False)
        retyped += 1
    return retyped


def main() -> int:
    args = parse_args()
    capture = LogCapture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    counters = instrument()

    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", args.text_encoder)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.LTXV,
    )
    load_seconds = time.perf_counter() - started

    model = clip.cond_stage_model
    formats = quant_summary(model)
    retyped, retype_dtype = 0, None
    if args.force_native and args.force_dequant:
        raise SystemExit("--force-native and --force-dequant are opposites; pick one")
    if args.force_native:
        retype_dtype = getattr(torch, args.native_dtype)
    elif args.force_dequant:
        # Generation runs in bf16, so float32 guarantees the mismatch cast_bias_weight reacts to.
        retype_dtype = torch.float32
    if retype_dtype is not None:
        retyped = retype_quantized(model, retype_dtype)

    tokens = clip.tokenize(args.prompt, skip_template=False, min_length=1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generated = clip.generate(
        tokens,
        do_sample=not args.greedy,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - started
    text = clip.decode(generated)

    report = {
        "text_encoder": args.text_encoder,
        "quant_formats": formats or {"none": 0},
        "modules_retyped": retyped,
        "retyped_to": str(retype_dtype) if retyped else None,
        "activation_precision": ("4-bit, native kernel" if counters["native_calls"] and not retyped
                                 else None),
        "prompt": args.prompt,
        "sampling": "greedy" if args.greedy else
                    f"T={args.temperature} top_k={args.top_k} top_p={args.top_p} "
                    f"min_p={args.min_p} rep={args.repetition_penalty} seed={args.seed}",
        "generated_text": text,
        "tokens_generated": len(generated),
        "load_seconds": round(load_seconds, 2),
        "generate_seconds": round(generate_seconds, 2),
        "tokens_per_second": round(len(generated) / generate_seconds, 2),
        "generate_peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "native_linear_calls": counters["native_calls"],
        "weight_dequant_calls": counters["dequant_calls"],
        "kernel_impls": sorted(counters["impls"]),
        "log_warnings": [m for level, m in capture.records if level in ("WARNING", "ERROR")],
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"{args.text_encoder}   {formats or 'unquantized'}")
    print(f"  {report['tokens_generated']} tokens in {report['generate_seconds']} s "
          f"= {report['tokens_per_second']} tok/s   "
          f"peak {report['generate_peak_vram_bytes'] / 1024**3:.2f} GiB")
    print(f"  native linear calls {report['native_linear_calls']}   "
          f"weight dequantizations {report['weight_dequant_calls']}")
    for impl in report["kernel_impls"]:
        print(f"  impl: {impl}")
    if retyped:
        print(f"  retyped {retyped} quantized weights to {retype_dtype}")
    print("-" * 78)
    print(text)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
