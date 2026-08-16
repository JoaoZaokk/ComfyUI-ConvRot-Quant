# ComfyUI ConvRot Quant

4-bit quantization for ComfyUI checkpoints that actually executes its kernel instead of quietly
dequantizing back to a full-precision GEMM — plus the measurements showing which 4-bit format is
worth using.

**Use `tools/quant_w4a8.py`. Do not use W4A4.** That is not a style preference; it is what the
numbers said.

![FP16 vs ConvRot W4A4 vs asym_w4a8_int8](docs/w4a4_vs_w4a8.png)

HunyuanVideo 1.5, RTX 3090, identical prompt, seed, steps, resolution, sampler and scheduler.
Model resident, seeds varied so ComfyUI could not serve a cached result. Times are ComfyUI's own
`Prompt executed`.

| Format | Time | VRAM staged | On disk | Image |
| --- | --- | --- | --- | --- |
| FP16 source | **2.94 s** | 15881 MB | 15.51 GiB | correct |
| ConvRot W4A4 | 4.85 s | 8113 MB | 7.92 GiB | **destroyed** |
| **asym_w4a8_int8** | **4.61 s** | 8437 MB | 8.24 GiB | **correct** |

ConvRot W4A4 is slower than FP16 *and* destroys the output, so its VRAM saving buys nothing. W4A8
is faster than W4A4 and produces a correct image for 4% more disk: **1.57x slower than FP16 for
1.88x less VRAM**, which is a real trade.

W4A4 was tested at Hadamard group sizes 256, 64 and 16. All three are unusable; the parameter does
not rescue it. The ConvRot paper reports 2.26x speedup on FLUX.1-dev; that did not reproduce here.

The W4A4 converter is kept because the format is a useful fixture for kernel and loader work, and
because a negative result with a reproduction is worth more than silence.

## Why 4-bit activations break a diffusion model

Both formats keep 4-bit **weights**. The one that works keeps 8-bit **activations**. That is the
axis that matters, and the likely reason is structural rather than numerical.

An autoregressive LLM ends every step with a discrete projection — argmax or a sample over a
vocabulary. A small perturbation in the logits usually selects the same token, so quantization
error is repeatedly snapped away, and the next step starts from an exactly-representable state. A
diffusion model has no such projection. The latent is continuous, and each denoising step feeds its
error straight into the next. Over several steps there is nothing to correct it.

W4A8 adds three defences on top of the same ConvRot rotation:

- activations stay at 8 bits, where Ampere's INT8 tensor-core path is mature and its INT4 path is not
- per-group scales (`group_size=16`) instead of one scale per row — on a `[8192, 2048]` layer that
  is 128 scales per row rather than 1
- a Lloyd-Max codebook, so the 16 int4 levels stop being uniformly spaced and are placed where the
  weights actually are

That last one is worth dwelling on: comfy-kitchen decides whether to build the codebook using a
**kurtosis probe** on the weight distribution. Uniform int4 levels assume the weights are evenly
spread; they are not, and the tails are exactly where the damage happens. Measuring that spread is
already part of the library.

## Components

- **`tools/quant_w4a8.py`** — converter for comfy-kitchen's `asym_w4a8_int8`. **Recommended.**
- **`tools/quant_w4a4.py`** — converter for `convrot_w4a4`. Kept as a fixture; see above.
- **`tools/verify_w4a4.py`** — metadata, layout, byte-for-byte source comparison and a real kernel run.
- **The `ConvRot W4A4 Native (Text Encoder)` node** — for text encoders, where stock ComfyUI stores
  quantized weights but never runs the kernel.
- **`compile_support.py`** — three runtime fixes that make `torch.compile` work at all.

Tested on an RTX 3090 (SM86) with ComfyUI `0.33.0`, comfy-kitchen `0.2.31`, Torch `2.13.0+cu130`.

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
git clone https://github.com/JoaoZaokk/ComfyUI-ConvRot-Quant ComfyUI/custom_nodes/ComfyUI-ConvRot-Quant
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

