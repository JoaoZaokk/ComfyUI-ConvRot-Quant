"""Make ConvRot W4A4 text encoders execute their native CUDA kernel.

Why this exists
---------------
ComfyUI runs text encoders in float32 on purpose: `comfy/sd1_clip.py` requests
`out_dtype=torch.float32` for the input embeddings and passes `dtype=torch.float32` into the
transformer, and `comfy/sd.py` matches that with `patcher.set_model_compute_dtype(torch.float32)`.

A quantized weight, however, is loaded with `orig_dtype` set to the module's compute dtype,
which for a Gemma/Qwen encoder is the dtype of `model.norm.weight` (BF16). So the activations
arrive as FP32 while the weight advertises BF16, and `comfy/ops.py` reacts to that mismatch by
dequantizing the weight before the matmul. The quantized kernel never runs.

`QuantizedTensor.to(dtype=...)` only rewrites `params.orig_dtype`; it does not touch the packed
data. Retyping the quantized weights to the dtype the activations actually arrive in removes the
mismatch, so `F.linear` dispatches to the ConvRot layout handler and the native kernel runs.

Measured on an RTX 3090 with Gemma 3 12B ConvRot W4A4 through `LTXAVTextEncoderLoader`:
336/336 layers dispatched to `comfy_kitchen.backends.cuda.convrot_w4a4_linear`, zero weight
dequantizations, and prompt encoding went from 2.354 s to 0.539 s.

No ComfyUI core file is modified.
"""

import logging

import torch

from comfy.quant_ops import QuantizedTensor

from . import compile_support

# Registering the ConvRot kernel as an opaque custom op is what lets torch.compile work at all;
# doing it here rather than by editing site-packages means a comfy-kitchen upgrade cannot revert it.
compile_support.apply()

NATIVE_FORMATS = {"convrot_w4a4"}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class ConvRotNativeTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "activation_dtype": (["float32", "bfloat16", "float16"],),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "advanced/quantization"
    DESCRIPTION = (
        "Retypes ConvRot W4A4 weights to the dtype the activations arrive in so the native CUDA "
        "kernel runs instead of dequantizing to a full-precision GEMM. Use float32 for text "
        "encoders, which ComfyUI upcasts to float32 by design."
    )

    def apply(self, clip, activation_dtype):
        target = DTYPES[activation_dtype]
        model = clip.cond_stage_model

        retyped = 0
        already = 0
        skipped_patched = 0
        for module in model.modules():
            if getattr(module, "quant_format", None) not in NATIVE_FORMATS:
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, QuantizedTensor):
                continue
            if len(getattr(module, "weight_function", [])) > 0:
                skipped_patched += 1
                continue
            if weight.dtype == target:
                already += 1
                continue
            module.weight = torch.nn.Parameter(weight.to(dtype=target), requires_grad=False)
            retyped += 1

        if retyped == 0 and already == 0:
            logging.warning("ConvRotNativeTextEncoder: no ConvRot W4A4 layers found in this CLIP")
        else:
            logging.info(
                "ConvRotNativeTextEncoder: {} layers retyped to {}, {} already matching".format(
                    retyped, activation_dtype, already
                )
            )
        if skipped_patched > 0:
            logging.warning(
                "ConvRotNativeTextEncoder: {} layers carry weight patches (LoRA); those keep "
                "dequantizing because the patch needs a dense weight".format(skipped_patched)
            )

        return (clip,)


NODE_CLASS_MAPPINGS = {"ConvRotNativeTextEncoder": ConvRotNativeTextEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {"ConvRotNativeTextEncoder": "ConvRot W4A4 Native (Text Encoder)"}
