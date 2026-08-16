"""Run a fixed set of verifiable questions through several checkpoints and score the answers.

One question is an anecdote. This exists because the single prime-number prompt that first showed
ConvRot W4A4 failing is not, on its own, enough to rank a fix against it -- a checkpoint can miss
one token and be broadly fine, or pass one question and be broadly broken.

Every question has an answer that is checkable by string match rather than by judgement, and
greedy decoding makes each run reproducible. The model loads once per checkpoint and answers all
of them. Native-kernel and dequantization counts are reported per checkpoint, because a run that
silently dequantized is not testing the format it claims to.

Scores are a floor, not a grade: a wrong-but-fluent answer counts as wrong, and a right answer
inside a rambling one counts as right.

    python tools/quality_battery.py --text-encoder a.safetensors --text-encoder b.safetensors
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from gemma_chat import QUANT_FORMATS, instrument, quant_summary, retype_quantized  # noqa: E402

# (label, prompt, predicate over the lowercased answer)
QUESTIONS = [
    ("primes", "List the first 8 prime numbers, separated by commas. Answer with the list only.",
     lambda t: re.sub(r"[^\d,]", "", t).strip(",").startswith("2,3,5,7,11,13,17,19")),
    ("product", "What is 47 multiplied by 89? Answer with the number only.",
     lambda t: "4183" in t.replace(",", "").replace(".", "")),
    ("capital", "What is the capital city of Australia? Answer with the city name only.",
     lambda t: "canberra" in t),
    ("bigger", "Which number is larger, 3.9 or 3.11? Answer with the number only.",
     lambda t: "3.9" in t and "3.11" not in t.replace("3.9", "")),
    ("letters", "How many times does the letter 'a' appear in the word 'banana'? "
                "Answer with the digit only.",
     lambda t: re.search(r"\b3\b", t) is not None),
    ("february", "How many days were there in February 2024? Answer with the number only.",
     lambda t: re.search(r"\b29\b", t) is not None),
    ("reverse", "Write the word 'stone' backwards. Answer with the word only.",
     lambda t: "enots" in t),
    ("plural", "Escreva o plural de 'cidadão'. Responda apenas com a palavra.",
     lambda t: "cidadãos" in t),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-encoder", action="append", default=[],
                        help="file name inside models/text_encoders, repeatable. Optional only "
                             "because a run may consist entirely of the other modes below.")
    parser.add_argument("--force-dequant", action="append", default=[],
                        help="checkpoint name to run with weights retyped so ComfyUI dequantizes "
                             "them, giving 4-bit weights with full-precision activations")
    parser.add_argument("--convrot-int8", action="append", default=[],
                        help="convrot_w4a4 checkpoint to run with linear_dtype='int8'. Same int4 "
                             "weights, but the kernel quantizes activations to int8 and takes the "
                             "int4-weight/int8-act GEMM path. Native, not emulated.")
    parser.add_argument("--emulate-a4", action="append", default=[],
                        help="checkpoint to run dequantized with an int4 activation round trip "
                             "injected before every quantized Linear. Emulation -- the only way to "
                             "put int4 activations in front of weights whose kernel has no A4 path.")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


_FORCE_INT8 = {"on": False}


def set_convrot_int8(enabled: bool) -> None:
    """Force every ConvRot call onto its int8-activation branch, once, at the tensor layer."""
    import comfy_kitchen.tensor.convrot_w4a4 as convrot

    if "original" not in _FORCE_INT8:
        original = convrot.convrot_w4a4_linear
        _FORCE_INT8["original"] = original

        def routed(*a, **kw):
            if _FORCE_INT8["on"]:
                kw["linear_dtype"] = "int8"
            return original(*a, **kw)

        convrot.convrot_w4a4_linear = routed
    _FORCE_INT8["on"] = enabled


def install_a4_emulation(model, convrot_groupsize: int = 256) -> int:
    """Inject ConvRot's exact activation quantization before each quantized Linear.

    rotate -> per-token absmax int4 -> dequantize -> rotate back. The rotation is its own
    inverse here, so in full precision this is the identity and everything it changes is the
    int4 round trip -- the same arithmetic convrot_w4a4_linear applies internally, made available
    to weight formats whose kernel has no int4-activation path.
    """
    from comfy_kitchen.backends.eager.convrot_w4a4 import (
        _build_hadamard, _rotate_activation, quantize_signed_int4_rowwise,
        _unpack_int4_row_major)

    installed = 0
    for module in model.modules():
        if getattr(module, "quant_format", None) not in QUANT_FORMATS:
            continue

        def hook(mod, inputs):
            x = inputs[0]
            if x.shape[-1] % convrot_groupsize:
                return None
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            h = _build_hadamard(convrot_groupsize, device=flat.device, dtype=flat.dtype)
            rotated = _rotate_activation(flat, h, convrot_groupsize).contiguous()
            packed, scale = quantize_signed_int4_rowwise(rotated)
            recon = _unpack_int4_row_major(packed).to(flat.dtype) * scale.to(flat.dtype).reshape(-1, 1)
            return (_rotate_activation(recon, h, convrot_groupsize).reshape(shape),) + inputs[1:]

        module.register_forward_pre_hook(hook)
        installed += 1
    return installed


def run_checkpoint(name: str, dequant: bool, args, counters, mode: str = "native") -> dict:
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", name)
    clip = comfy.sd.load_clip(ckpt_paths=[path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.LTXV)
    model = clip.cond_stage_model
    formats = quant_summary(model)
    retyped = retype_quantized(model, torch.float32) if dequant else 0
    emulated = install_a4_emulation(model) if mode == "emulate-a4" else 0

    counters["native_calls"] = counters["dequant_calls"] = 0
    counters["impls"] = set()
    answers, passed = [], 0
    started = time.perf_counter()
    for label, prompt, check in QUESTIONS:
        tokens = clip.tokenize(prompt, skip_template=False, min_length=1)
        text = clip.decode(clip.generate(tokens, do_sample=False, max_length=args.max_length,
                                         temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
                                         repetition_penalty=1.0, seed=0)).strip()
        ok = bool(check(text.lower()))
        passed += ok
        answers.append({"label": label, "ok": ok, "answer": " ".join(text.split())[:160]})
        print(f"  {'OK ' if ok else 'X  '} {label:<10} {' '.join(text.split())[:96]}", flush=True)
    elapsed = time.perf_counter() - started

    result = {
        "text_encoder": name,
        "mode": mode if mode != "native" else ("force-dequant (W4A16)" if dequant else "native"),
        "a4_hooks_installed": emulated,
        "quant_formats": formats or {"none": 0},
        "modules_retyped": retyped,
        "passed": passed, "total": len(QUESTIONS),
        "seconds": round(elapsed, 1),
        "native_linear_calls": counters["native_calls"],
        "weight_dequant_calls": counters["dequant_calls"],
        "kernel_impls": sorted(counters["impls"]),
        "answers": answers,
    }
    del clip, model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    counters = instrument()
    set_convrot_int8(False)

    # (checkpoint, dequantize weights?, mode) -- one row of the weight-format x activation-precision
    # matrix per entry, and the mode decides which activation precision that row runs at.
    runs = [(name, False, "native") for name in args.text_encoder]
    runs += [(name, True, "native") for name in args.force_dequant]
    runs += [(name, False, "convrot-int8") for name in args.convrot_int8]
    runs += [(name, True, "emulate-a4") for name in args.emulate_a4]
    if not runs:
        raise SystemExit("Nothing to run: pass at least one of --text-encoder, --force-dequant, "
                         "--convrot-int8 or --emulate-a4")
    tags = {"native": "", "convrot-int8": "   [linear_dtype=int8 -> A8]",
            "emulate-a4": "   [dequantized weights + emulated A4]"}
    results = []
    for name, dequant, mode in runs:
        suffix = "   [force-dequant -> W4A16]" if dequant and mode == "native" else tags[mode]
        label = f"{name}{suffix}"
        print(f"\n{'=' * 78}\n{label}\n{'-' * 78}", flush=True)
        set_convrot_int8(mode == "convrot-int8")
        result = run_checkpoint(name, dequant, args, counters, mode)
        results.append(result)
        print(f"  -> {result['passed']}/{result['total']}   "
              f"native {result['native_linear_calls']}  dequant {result['weight_dequant_calls']}",
              flush=True)

    print(f"\n{'=' * 78}")
    print(f"{'checkpoint':<52}{'score':>8}{'kernel':>16}")
    for result in results:
        kernel = ("dequant" if result["weight_dequant_calls"] else
                  "native" if result["native_linear_calls"] else "unquantized")
        tag = result["text_encoder"][:40] + (" [A16]" if result["modules_retyped"] else "")
        print(f"{tag:<52}{result['passed']}/{result['total']:<6}{kernel:>16}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
