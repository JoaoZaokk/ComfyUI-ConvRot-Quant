# ComfyUI ConvRot W4A4

Convert high-precision ComfyUI checkpoints to ConvRot W4A4 and make them actually execute the
native INT4 tensor-core kernel instead of quietly dequantizing back to a full-precision GEMM.

Two pieces:

- **`tools/quant_w4a4.py`** — a converter that writes ComfyUI-native `convrot_w4a4` safetensors.
- **The `ConvRot W4A4 Native (Text Encoder)` node** — a fix for text encoders, where stock
  ComfyUI stores the quantized weights but never runs the kernel.

Tested on an RTX 3090 (SM86) with ComfyUI `0.29.0`, comfy-kitchen `0.2.23`, Torch `2.12.1+cu130`.

## The problem this solves

A checkpoint can be perfectly quantized and still run at full precision. ComfyUI decides per
layer, at forward time, whether to dispatch the quantized kernel — and for text encoders it never
does.

ComfyUI runs text encoders in FP32 on purpose:

- `comfy/sd1_clip.py` requests the input embeddings with `out_dtype=torch.float32`, passes
  `dtype=torch.float32` into the transformer, and returns `.float()` outputs.
- `comfy/sd.py` matches that with `patcher.set_model_compute_dtype(torch.float32)`.

But a quantized weight is loaded with its `orig_dtype` set to the module's *compute* dtype, which
for a Gemma/Qwen encoder is the dtype of `model.norm.weight` — BF16. So activations arrive as FP32
while the weight advertises BF16, and `comfy/ops.py` reacts to the mismatch:

```python
if weight_has_function or weight.dtype != dtype:
    weight = weight.to(dtype=dtype)
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()      # <- the kernel never runs
```

The result is `W4 storage -> dequantize -> BF16 GEMM`. This affects every quantized text encoder
format, not just ConvRot: `float8_e4m3fn`, `int8_tensorwise` and `convrot_w4a4` all behave this way.

The FP32 policy is deliberate and should not be removed — it would change numerics for every
encoder. The *mismatch* is the bug. And `QuantizedTensor.to(dtype=...)` only rewrites
`params.orig_dtype`; it never touches the packed data. So retyping the quantized weights to the
dtype the activations actually arrive in removes the mismatch at zero cost, and `F.linear`
dispatches to the ConvRot layout handler.

That is all the node does. **No ComfyUI core file is modified**, so updates cannot clobber it.

### Measured

Gemma 3 12B ConvRot W4A4 through `LTXAVTextEncoderLoader`, RTX 3090, one load, two encodes:

| | Stock | With the node |
| --- | --- | --- |
| `convrot_w4a4_linear` calls | 0 | **336** |
| Weight dequantizations | 336 | **0** |
| Backend | none | `comfy_kitchen.backends.cuda.convrot_w4a4_linear` |
| Prompt encode | 2.354 s | **0.539 s** |

Diffusion models do **not** need the node: `pick_operations` builds their ops with
`full_precision_mm=False` and matching activations, and `convrot_w4a4` is never added to the
`disabled` set, so they dispatch natively on their own.

## torch.compile support

Importing this package also repairs `torch.compile` for ConvRot W4A4, applied at runtime so a
`pip install --upgrade comfy-kitchen` cannot silently revert it. Three independent defects, each
verified separately and then together:

| Defect | Symptom | Fix |
| --- | --- | --- |
| `convrot_w4a4_linear` is not an opaque custom op | `RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor...)` — Dynamo traces into the CUDA kernel | `torch.library.custom_op` + `register_fake` |
| `__tensor_unflatten__` drops `outer_size`/`outer_stride` | crash when input length changes between runs | adopt `outer_size`, forward the stride |
| no `_stable_hash_for_caching` | a cached compiled artifact is reused across incompatible graphs, then `AttributeError: '_OpNamespace' ... has no attribute` | stable hash over layout, dtypes, shapes and non-tensor params |

Measured on an RTX 3090, HunyuanVideo 1.5 ConvRot W4A4, with `site-packages` left pristine:
compiled output `max_abs_diff 0.00000000` against eager, and a second call at a different sequence
length also `0.00000000`.

Note the first two are separate holes: **neither one alone makes `torch.compile` work.** Tested
individually before being tested together. comfy-kitchen already registers ~37 other ops with
`torch.library.custom_op` (`comfy_kitchen::adaln`, `::int8_linear`, `::scaled_mm_svdquant_w4a4`);
ConvRot's linear was simply never added. The dynamic-shape half mirrors
[comfy-kitchen PR #52](https://github.com/Comfy-Org/comfy-kitchen/pull/52), open since 2026-06-25.

A single quantized Linear compiles to one opaque op, so expect no speedup from `torch.compile` at
that granularity — the gain comes from fusing the surrounding norms, activations and RoPE across a
whole model.

## Install

Clone into your ComfyUI `custom_nodes` directory and restart:

```bash
git clone https://github.com/JoaoZaokk/ComfyUI-ConvRot-W4A4 ComfyUI/custom_nodes/ComfyUI-ConvRot-W4A4
```

No dependencies beyond what ComfyUI already ships. The converter additionally uses `psutil`,
which ComfyUI already requires.

## Using the node

Add **ConvRot W4A4 Native (Text Encoder)** between your text-encoder loader and
`CLIPTextEncode`. Set `activation_dtype` to `float32` — that is what ComfyUI upcasts text encoders
to.

```
LTXAVTextEncoderLoader ──▶ ConvRot W4A4 Native ──▶ CLIPTextEncode
```

It logs how many layers it retyped. Layers carrying weight patches (LoRA) are skipped and counted
separately, because a patch needs a dense weight and will keep dequantizing.

## Using the converter

```bash
python tools/quant_w4a4.py --input path/to/model.safetensors --dry-run
python tools/quant_w4a4.py --input path/to/model.safetensors
```

Output lands next to the source as `<name>_w4a4_convrot.safetensors` with a `.quant.json` sidecar
recording source, sizes, architecture, layout, backend, excluded patterns, and versions.

Verify before trusting it:

```bash
python tools/verify_w4a4.py out_w4a4_convrot.safetensors --source source.safetensors --kernel-smoke
```

This checks the metadata and per-layer layout, compares **every preserved tensor byte for byte**
against the source, confirms normal ComfyUI resolves the CUDA backend, and runs one real layer
through the kernel.

### Supported profiles

| Profile | Quantized | Preserved |
| --- | --- | --- |
| `gemma`, `qwen` | `model.layers.N.self_attn.{q,k,v,o}_proj`, `model.layers.N.mlp.{gate,up,down}_proj` | embeddings, norms, `lm_head`, vision tower |
| `hunyuan_video_15` | `double_blocks.N.{img,txt}_attn_{qkv,proj}`, `double_blocks.N.{img,txt}_mlp.fc{1,2}` | `*_mod.linear` (adaLN), norms, `img_in`, `txt_in`, `byt5_in`, `vision_in`, `time_in`, `final_layer`, embeddings, all biases |

`hunyuan_video_15` is detected structurally, mirroring the HunyuanVideo branch of
`comfy.model_detection`, not from the file name. Profiles are strict allowlists — do not point one
at an architecture it was not derived from.

## Design notes worth knowing before adding a profile

**Layer names in the metadata must use the checkpoint's own key convention, not ComfyUI's module
names.** `comfy.utils.convert_old_quants` injects `<layer>.comfy_quant` keys *before*
`process_unet_state_dict` runs, and that function remaps by substring
(`"_attn_qkv." -> "_attn.qkv."`, `"mlp.fc1." -> "mlp.0."`, ...), so it carries the injected
`.comfy_quant` and `.weight_scale` keys along with the weights. Naming layers after the module
paths would break this.

**Group sizes.** `convrot_groupsize` 256, `quant_group_size` 64. Layer selection requires
`shape[1] % 256 == 0`. These match what `comfy/ops.py` assumes when it rebuilds
`TensorCoreConvRotW4A4Layout.Params`.

**Streaming, never mmap.** Output header offsets are computed up front, quantized layers are read
by byte range and written one at a time, and preserved tensors are copied in 16 MiB chunks.
Mapping a 21.9 GiB source on Windows produced `os error 1455` and twice crashed `torch_cpu.dll`
with `0xC0000005`.

**Native-backend preflight.** Before touching a tensor the converter spawns a subprocess that
imports `comfy.quant_ops` and asks the comfy-kitchen registry which implementation would be
selected. If it is not `comfy_kitchen.backends.cuda.*`, the conversion hard-refuses rather than
producing a checkpoint that can only ever run dequantized.

## Caveats

- Quantization error is real. Per-layer relative RMSE on random inputs is roughly 0.20–0.23. Run a
  matched-parameter A/B before using a converted model for anything you care about.
- `CLIP.clone()` shares `cond_stage_model`, so the node's retype is visible to every clone of that
  CLIP. Harmless — it changes only the declared logical dtype — but worth knowing.
- The converter refuses to overwrite a source, an existing output, an existing sidecar, or a stale
  `.partial`, and refuses any checkpoint that already carries quantization metadata.

## License

MIT. See [LICENSE](LICENSE).
