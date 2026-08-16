"""Make ConvRot W4A4 survive torch.compile, patched at runtime.

Two independent defects block `torch.compile` on a ConvRot W4A4 model. Neither fixes the other;
both were verified by measurement, and each was tested alone before being tested together.

1. `comfy_kitchen.tensor.convrot_w4a4.convrot_w4a4_linear` is not registered as an opaque custom
   op, so Dynamo traces *into* the CUDA kernel and dies on a FakeTensor data pointer:

       RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor).
       ... please wrap the custom kernel into an opaque custom op.

   Fixed here with `torch.library.custom_op` plus a fake (meta) implementation, which is the fix
   PyTorch's own error message prescribes. comfy-kitchen already does exactly this for ~37 other
   ops (`comfy_kitchen::adaln`, `comfy_kitchen::int8_linear`, `comfy_kitchen::scaled_mm_svdquant_w4a4`
   and so on) -- ConvRot's linear was simply never added.

2. `QuantizedTensor.__tensor_unflatten__` ignores the `outer_size`/`outer_stride` that torch.compile
   passes under dynamic shapes, so a model crashes when the input size changes between runs -- for
   example when prompt length varies. This mirrors the fix proposed upstream in
   Comfy-Org/comfy-kitchen PR #52, which has been open since 2026-06-25.

Applying both at runtime rather than editing site-packages means a `pip install --upgrade
comfy-kitchen` cannot silently revert them.

Import is side-effect-free if comfy_kitchen is missing or already patched.
"""

from __future__ import annotations

import dataclasses
import logging

import torch

_state = {"custom_op": False, "dynamic_shapes": False, "stable_hash": False}


def _install_stable_hash() -> bool:
    """Give QuantizedTensor the stable hash PT2 asks for, so cached graphs cannot be mismatched.

    Without it torch warns "QuantizedTensor does not implement _stable_hash_for_caching", and the
    consequence is real: a compiled artifact cached from one process was reused in another whose
    custom op namespace differed, producing

        AttributeError: '_OpNamespace' 'convrot_poc' object has no attribute 'w4a4_linear'

    The hash covers everything that changes the generated graph: layout, storage and logical dtype,
    shapes, and the non-tensor layout parameters (group sizes, linear_dtype, transposed).
    """
    import comfy_kitchen.tensor.base as base

    quantized_tensor = base.QuantizedTensor
    if hasattr(quantized_tensor, "_stable_hash_for_caching"):
        return False

    def _stable_hash_for_caching(self) -> str:
        params = self._params
        parts = [
            self._layout_cls,
            str(self._qdata.dtype),
            str(tuple(self._qdata.shape)),
            str(params.orig_dtype),
            str(tuple(params.orig_shape)),
        ]
        for field in dataclasses.fields(params):
            value = getattr(params, field.name)
            if not isinstance(value, torch.Tensor):
                parts.append(f"{field.name}={value}")
        return "|".join(parts)

    quantized_tensor._stable_hash_for_caching = _stable_hash_for_caching
    return True


def _install_custom_op() -> bool:
    """Route convrot_w4a4_linear through an opaque custom op so Dynamo stops tracing into it."""
    import comfy_kitchen.tensor.convrot_w4a4 as convrot

    if getattr(convrot, "_convrot_custom_op_installed", False):
        return False

    original = convrot.convrot_w4a4_linear

    @torch.library.custom_op("comfy_convrot_w4a4::linear", mutates_args=())
    def _linear(
        x: torch.Tensor,
        qweight: torch.Tensor,
        wscales: torch.Tensor,
        bias: torch.Tensor | None,
        convrot_groupsize: int,
        quant_group_size: int,
        linear_dtype: str,
    ) -> torch.Tensor:
        return original(
            x, qweight, wscales, bias=bias,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )

    @_linear.register_fake
    def _(x, qweight, wscales, bias, convrot_groupsize, quant_group_size, linear_dtype):
        # qweight is [out_features, in_features // 2]: int8 storage holding two signed int4 each.
        return x.new_empty((*x.shape[:-1], qweight.shape[0]))

    def dispatching(
        x, qweight, wscales, bias=None,
        convrot_groupsize=256, quant_group_size=64, linear_dtype="int4",
    ):
        return torch.ops.comfy_convrot_w4a4.linear(
            x, qweight, wscales, bias,
            int(convrot_groupsize), int(quant_group_size), str(linear_dtype),
        )

    convrot.convrot_w4a4_linear = dispatching
    convrot._convrot_custom_op_installed = True
    return True


def _install_dynamic_shapes() -> bool:
    """Honour the outer_size/outer_stride torch.compile passes when rebuilding the subclass."""
    import comfy_kitchen.tensor.base as base

    quantized_tensor = base.QuantizedTensor
    if getattr(quantized_tensor, "_convrot_dynshape_patched", False):
        return False

    def __new__(cls, qdata, layout_cls, params, outer_stride=None):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            params.orig_shape,
            strides=outer_stride,
            device=qdata.device,
            dtype=params.orig_dtype,
            requires_grad=False,
        )

    def __init__(self, qdata, layout_cls, params, outer_stride=None):
        assert isinstance(layout_cls, str)
        self._qdata = qdata
        self._layout_cls = layout_cls
        self._params = params

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        params_kwargs = dict(ctx["non_tensor_fields"])
        for field_name, attr_name in ctx["tensor_fields"].items():
            params_kwargs[field_name] = inner_tensors[attr_name]
        params = ctx["params_class"](**params_kwargs)
        # torch.compile passes real sequences that the rebuilt subclass must report exactly
        # (asserted in torch._subclasses.meta_utils). Other callers, such as
        # comfy.memory_management.interpret_gathered_like, pass scalar 0 placeholders meaning
        # "keep the stored shape", so only act on real sequences.
        if isinstance(outer_size, (tuple, list, torch.Size)):
            params = dataclasses.replace(params, orig_shape=tuple(outer_size))
            return quantized_tensor(
                inner_tensors["_qdata"], ctx["layout_cls"], params, outer_stride=outer_stride,
            )
        return quantized_tensor(inner_tensors["_qdata"], ctx["layout_cls"], params)

    quantized_tensor.__new__ = __new__
    quantized_tensor.__init__ = __init__
    quantized_tensor.__tensor_unflatten__ = __tensor_unflatten__
    quantized_tensor._convrot_dynshape_patched = True
    return True


def apply() -> dict:
    """Apply both patches. Safe to call more than once; never raises."""
    try:
        import comfy_kitchen  # noqa: F401
    except Exception as error:
        logging.info(f"ConvRot W4A4: comfy_kitchen unavailable, compile support skipped ({error})")
        return dict(_state)

    installers = (
        ("custom_op", _install_custom_op),
        ("dynamic_shapes", _install_dynamic_shapes),
        ("stable_hash", _install_stable_hash),
    )
    for name, installer in installers:
        if _state[name]:
            continue
        try:
            _state[name] = installer()
        except Exception as error:
            logging.warning(f"ConvRot W4A4: could not install {name} compile patch: {error}")

    applied = [name for name, done in _state.items() if done]
    if applied:
        logging.info(f"ConvRot W4A4: torch.compile support installed ({', '.join(applied)})")
    return dict(_state)
