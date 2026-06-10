#!/usr/bin/env python3
"""Weight Quantization Utilities: RTN and AWQ.

Provides:
- RTN: W8 (per-channel symmetric) and W4 (per-group g=128 symmetric)
- AWQ: Activation-aware W4 quantization with per-layer sequential optimization
  following the autoawq algorithm: weight-absorb scaling + module-level MSE

AWQ pipeline for RL self-speculative decoding:
  1. Capture per-layer calibration inputs (sequential layer-by-layer forward)
  2. For each layer: find optimal scales via grid search on module output MSE
  3. Absorb scales into LayerNorm/Linear weights (no runtime compensation needed)
  4. Quantize with standard RTN (QuantizedLinear)

In RL context, calibration data = rollout outputs from current policy.
This enables dynamic requantization as the model evolves during training.

NOTE: This is a simulator utility, NOT a production quantization kernel.
QuantizedLinear stores weights as int8 and dequantizes to FP16 for F.linear().
Latency projections come from the roofline model separately.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── QuantizedLinear & RTN ───────────────────────────────────────────


class QuantizedLinear(nn.Module):
    """Linear layer with RTN-quantized weights.

    Stores quantized weights (int8) and scales (fp16).
    Forward: dequantize to fp16 → F.linear().
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
        group_size: int = -1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.register_buffer(
            "int_weight", torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        if group_size <= 0:
            self.register_buffer(
                "scales", torch.zeros(out_features, 1, dtype=torch.float16)
            )
        else:
            num_groups = (in_features + group_size - 1) // group_size
            self.register_buffer(
                "scales", torch.zeros(out_features, num_groups, dtype=torch.float16)
            )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp16 = self._dequantize()
        return F.linear(x, weight_fp16, self.bias)

    def _dequantize(self) -> torch.Tensor:
        if self.group_size <= 0:
            return self.int_weight.to(self.scales.dtype) * self.scales
        else:
            out_features, in_features = self.int_weight.shape
            num_groups = self.scales.shape[1]
            padded_in = num_groups * self.group_size
            if padded_in > in_features:
                int_w = F.pad(self.int_weight, (0, padded_in - in_features))
            else:
                int_w = self.int_weight
            int_w = int_w.reshape(out_features, num_groups, self.group_size)
            w_fp16 = int_w.to(self.scales.dtype) * self.scales.unsqueeze(-1)
            w_fp16 = w_fp16.reshape(out_features, padded_in)
            if padded_in > in_features:
                w_fp16 = w_fp16[:, :in_features]
            return w_fp16


def quantize_model_rtn(
    model: nn.Module,
    bits: int = 8,
    group_size: int = -1,
    skip_modules: set[str] | None = None,
) -> nn.Module:
    """Apply RTN quantization to all Linear layers in-place."""
    if bits == 8:
        group_size = -1
        qmin, qmax = -128, 127
    elif bits == 4:
        if group_size <= 0:
            group_size = 128
        qmin, qmax = -8, 7
    else:
        raise ValueError(f"Unsupported bits: {bits}. Use 4 or 8.")

    if skip_modules is None:
        skip_modules = set()

    replacements = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in skip_modules:
            continue

        w = module.weight.data
        out_features, in_features = w.shape
        has_bias = module.bias is not None

        ql = QuantizedLinear(
            in_features, out_features, bias=has_bias,
            bits=bits, group_size=group_size,
        )

        if group_size <= 0:
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
            int_w = (w / scale).round().clamp(qmin, qmax).to(torch.int8)
            ql.int_weight.copy_(int_w)
            ql.scales.copy_(scale.to(torch.float16))
        else:
            num_groups = (in_features + group_size - 1) // group_size
            padded_in = num_groups * group_size
            if padded_in > in_features:
                w_padded = F.pad(w, (0, padded_in - in_features))
            else:
                w_padded = w
            w_groups = w_padded.reshape(out_features, num_groups, group_size)
            scale = w_groups.abs().amax(dim=2).clamp(min=1e-10) / qmax
            int_w = (w_groups / scale.unsqueeze(-1)).round().clamp(qmin, qmax).to(torch.int8)
            int_w = int_w.reshape(out_features, padded_in)
            if padded_in > in_features:
                int_w = int_w[:, :in_features]
            ql.int_weight.copy_(int_w)
            ql.scales.copy_(scale.to(torch.float16))

        if has_bias:
            ql.bias.copy_(module.bias.data.to(torch.float16))

        replacements[name] = ql

    for name, ql in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], ql.to(w.device))

    return model


def retain_fp16_lm_head(
    quantized_model: nn.Module, fp16_model: nn.Module
) -> None:
    """Replace quantized lm_head with FP16 copy from target model."""
    if hasattr(quantized_model, "lm_head") and hasattr(fp16_model, "lm_head"):
        device = next(quantized_model.parameters()).device
        fp16_lm = fp16_model.lm_head
        quantized_model.lm_head = nn.Linear(
            fp16_lm.in_features, fp16_lm.out_features,
            bias=fp16_lm.bias is not None,
            device=device, dtype=torch.float16,
        )
        quantized_model.lm_head.weight.data.copy_(fp16_lm.weight.data.to(torch.float16))
        if fp16_lm.bias is not None:
            quantized_model.lm_head.bias.data.copy_(fp16_lm.bias.data.to(torch.float16))


# ─── AWQ: Activation-Aware Weight Quantization (autoawq algorithm) ──


def _pseudo_quantize_tensor(
    w: torch.Tensor, bits: int = 4, group_size: int = 128,
) -> torch.Tensor:
    """Simulate quantize → dequantize for error measurement."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    org_shape = w.shape
    if group_size > 0:
        w = w.reshape(-1, group_size)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
    w_q = (w / scale).round().clamp(qmin, qmax) * scale
    return w_q.reshape(org_shape)


def _get_qwen2_scaling_pairs(layer: nn.Module, input_feat: dict, module_kwargs: dict):
    """Define the 4 AWQ scaling pairs for Qwen2-style decoder layers.

    Following autoawq/models/qwen2.py::get_layers_for_scaling():
    1. input_layernorm → [q_proj, k_proj, v_proj]  (inspect: self_attn)
    2. v_proj → [o_proj]
    3. post_attention_layernorm → [gate_proj, up_proj]  (inspect: mlp)
    4. up_proj → [down_proj]

    For module2inspect forward calls:
    - self_attn needs (hidden_states, position_embeddings=...)
    - mlp needs just (x)
    - single linear needs just (x) via F.linear wrapper
    """
    pairs = []

    # Config 1: input_layernorm → Q, K, V
    # Qwen2Attention.forward(hidden_states, position_embeddings, attention_mask, ...)
    # All three are required positional args. We wrap self_attn to handle this.
    pos_emb = module_kwargs.get("position_embeddings")

    class _AttnWrapper(nn.Module):
        """Wraps self_attn to match (hidden_states, **kwargs) calling convention."""
        def __init__(self, attn, pos_emb):
            super().__init__()
            self.attn = attn
            self.pos_emb = pos_emb
        def forward(self, x, **kwargs):
            return self.attn(x, position_embeddings=self.pos_emb, attention_mask=None)

    pairs.append({
        "prev_op": layer.input_layernorm,
        "linears": [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        "inp": input_feat.get("self_attn.q_proj"),
        "module2inspect": _AttnWrapper(layer.self_attn, pos_emb) if pos_emb is not None else layer.self_attn,
        "kwargs": {},
    })

    # Config 2: v_proj → o_proj (Linear → Linear)
    # Use o_proj directly as module2inspect (just a linear layer, takes x)
    if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
        pairs.append({
            "prev_op": layer.self_attn.v_proj,
            "linears": [layer.self_attn.o_proj],
            "inp": input_feat.get("self_attn.o_proj"),
            "module2inspect": layer.self_attn.o_proj,
            "kwargs": {},
        })

    # Config 3: post_attention_layernorm → gate, up (measure via mlp)
    # mlp.forward(x) — no extra kwargs
    pairs.append({
        "prev_op": layer.post_attention_layernorm,
        "linears": [layer.mlp.gate_proj, layer.mlp.up_proj],
        "inp": input_feat.get("mlp.gate_proj"),
        "module2inspect": layer.mlp,
        "kwargs": {},
    })

    # Config 4: up_proj → down_proj (Linear → Linear)
    pairs.append({
        "prev_op": layer.mlp.up_proj,
        "linears": [layer.mlp.down_proj],
        "inp": input_feat.get("mlp.down_proj"),
        "module2inspect": layer.mlp.down_proj,
        "kwargs": {},
    })

    return pairs


def _apply_awq_scale(prev_op: nn.Module, linears: list[nn.Linear], scales: torch.Tensor):
    """Absorb AWQ scales into prev_op and linear weights.

    Following autoawq/quantize/scale.py::apply_scale():
    - If prev_op is LayerNorm/RMSNorm: prev_op.weight /= s, fc.weight *= s
    - If prev_op is Linear: prev_op.weight /= s (output dim), fc.weight *= s (input dim)

    After this, no runtime compensation is needed — the scaling is baked into weights.
    """
    device = scales.device
    scales = scales.to(device)

    is_norm = hasattr(prev_op, "weight") and prev_op.weight.dim() == 1
    is_linear = isinstance(prev_op, nn.Linear)

    if is_norm:
        # LayerNorm/RMSNorm: weight is [hidden_dim]
        prev_op.weight.data.div_(scales.to(prev_op.weight.dtype))
        if hasattr(prev_op, "bias") and prev_op.bias is not None:
            prev_op.bias.data.div_(scales.to(prev_op.bias.dtype))
    elif is_linear:
        # Linear: scale output channels (dim 0)
        # prev_op.weight is [out, in], we scale the output dimension
        scales_out = scales.view(-1, 1).to(prev_op.weight.dtype)
        prev_op.weight.data.div_(scales_out)
        if prev_op.bias is not None:
            prev_op.bias.data.div_(scales.to(prev_op.bias.dtype))

    # Scale all downstream linear layers' input dimension
    for fc in linears:
        fc.weight.data.mul_(scales.view(1, -1).to(fc.weight.dtype))


def _search_best_scale(
    module2inspect: nn.Module,
    linears: list[nn.Linear],
    x_samples: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
    kwargs: dict | None = None,
) -> torch.Tensor:
    """Find optimal AWQ scales via module-level output MSE grid search.

    Matches autoawq's _compute_best_scale():
    1. scales = x_mean^ratio, normalized
    2. For each ratio: W *= s, Q(W), W /= s → measure output MSE
    3. Pick ratio with lowest MSE

    The key: we measure ACTUAL module output error, not per-weight error.
    """
    if kwargs is None:
        kwargs = {}
    # Remove use_cache to avoid KV cache issues during calibration
    kwargs = {k: v for k, v in kwargs.items() if k not in ("use_cache", "past_key_values")}

    device = next(module2inspect.parameters()).device
    x_samples = x_samples.to(device)

    # x_mean: per-channel activation magnitude
    x_flat = x_samples.reshape(-1, x_samples.shape[-1])
    x_mean = x_flat.abs().float().mean(dim=0).clamp(min=1e-8)

    # w_mean: per-channel weight magnitude (normalized within groups)
    weight = torch.cat([m.weight.data for m in linears], dim=0)
    org_shape = weight.shape
    w_grouped = weight.reshape(-1, group_size).float()
    w_scale = w_grouped.abs() / (w_grouped.abs().amax(dim=1, keepdim=True) + 1e-6)
    w_mean = w_scale.reshape(org_shape).mean(dim=0)
    del weight, w_grouped, w_scale

    # Helper to call module with correct args
    def _forward(mod, x):
        try:
            out = mod(x, **kwargs)
        except TypeError:
            # Fallback: just use F.linear for single Linear modules
            if isinstance(mod, nn.Linear):
                out = F.linear(x, mod.weight, mod.bias)
            else:
                raise
        if isinstance(out, tuple):
            out = out[0]
        return out

    # FP16 reference output
    with torch.no_grad():
        fp16_output = _forward(module2inspect, x_samples).float()

    # Save original weights
    org_sd = {m: m.weight.data.clone() for m in linears}

    best_error = float("inf")
    best_scales = torch.ones_like(x_mean)

    for i in range(n_grid):
        ratio = i / n_grid
        scales = x_mean.pow(ratio).clamp(min=1e-4)
        scales = scales / (scales.max() * scales.min()).sqrt()
        scales[torch.isinf(scales)] = 1
        scales[torch.isnan(scales)] = 1
        scales_view = scales.view(1, -1).to(device)

        # Apply: W *= s, pseudo-quantize, W = Q(W*s) / s
        for m in linears:
            m.weight.data = org_sd[m].clone()
            m.weight.data.mul_(scales_view.to(m.weight.dtype))
            m.weight.data = _pseudo_quantize_tensor(
                m.weight.data, bits, group_size
            ) / scales_view.to(m.weight.dtype)

        # Measure output MSE
        with torch.no_grad():
            q_output = _forward(module2inspect, x_samples).float()

        loss = (fp16_output - q_output).pow(2).mean().item()

        if loss < best_error:
            best_error = loss
            best_scales = scales.clone()

    # Restore original weights
    for m in linears:
        m.weight.data = org_sd[m]

    return best_scales


def _collect_layer_input_feat(
    layer: nn.Module, calib_input: torch.Tensor, module_kwargs: dict,
) -> dict[str, torch.Tensor]:
    """Collect input activations for all Linear layers in a decoder layer.

    Runs a single forward pass through the layer, capturing inputs via hooks.
    Returns dict mapping relative name → concatenated input tensor.
    """
    input_feat = defaultdict(list)
    hooks = []

    named_linears = {
        name: mod for name, mod in layer.named_modules()
        if isinstance(mod, nn.Linear)
    }

    for name, mod in named_linears.items():
        def make_hook(n):
            def hook_fn(module, args, output):
                x = args[0]
                input_feat[n].append(x.detach().cpu())
            return hook_fn
        hooks.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        layer_output = layer(calib_input, **module_kwargs)

    for h in hooks:
        h.remove()

    # Concatenate along batch dim
    for k in input_feat:
        input_feat[k] = torch.cat(input_feat[k], dim=0)

    # Return layer output for next layer's calibration
    if isinstance(layer_output, tuple):
        layer_output = layer_output[0]

    return dict(input_feat), layer_output


def awq_quantize_from_calibration(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    skip_modules: set[str] | None = None,
    n_grid: int = 20,
) -> nn.Module:
    """Full AWQ pipeline following the autoawq algorithm.

    Per-layer sequential processing:
    1. Forward calibration data through embedding → get initial hidden states
    2. For each decoder layer:
       a. Collect per-linear input activations
       b. For each of 4 scaling pairs: grid search → absorb scales into weights
       c. Quantize all linears with RTN (QuantizedLinear)
       d. Forward calibration data through (now quantized) layer → update for next layer
    3. Quantize LM head

    This matches autoawq's AwqQuantizer.quantize() algorithm:
    - Scales are absorbed into LayerNorm/Linear weights (no runtime x*s)
    - Module-level output MSE guides scale selection
    - Sequential layer processing propagates quantization effects correctly
    """
    if skip_modules is None:
        skip_modules = set()

    device = next(model.parameters()).device
    model.eval()
    num_layers = len(model.model.layers)

    print(f"  AWQ: {num_layers} layers, {len(calibration_data)} calibration samples, W{bits} g={group_size}")

    # Step 1: Forward calibration data through embedding to get initial hidden states
    print(f"  AWQ: Capturing initial hidden states...")
    all_hidden = []
    for input_ids in calibration_data:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            hidden = model.model.embed_tokens(input_ids)
        all_hidden.append(hidden.cpu())

    # Concatenate calibration hidden states: [total_samples, max_seq_len, hidden_dim]
    # Pad to same length
    max_len = max(h.shape[1] for h in all_hidden)
    hidden_dim = all_hidden[0].shape[2]
    padded = torch.zeros(len(all_hidden), max_len, hidden_dim, dtype=all_hidden[0].dtype)
    for i, h in enumerate(all_hidden):
        padded[i, :h.shape[1], :] = h
    calib_hidden = padded.to(device)

    # Prepare module_kwargs for decoder layers
    # Qwen2 needs position_ids and position_embeddings
    batch_size = calib_hidden.shape[0]
    position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    module_kwargs = {"position_ids": position_ids}

    # Compute position embeddings if transformers >= 4.48
    if hasattr(model.model, "rotary_emb"):
        with torch.no_grad():
            cos, sin = model.model.rotary_emb(calib_hidden, position_ids)
            module_kwargs["position_embeddings"] = (cos, sin)

    # Step 2: Process each layer sequentially
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        print(f"  AWQ: Layer {layer_idx}/{num_layers-1}", end="", flush=True)

        # 2a. Collect input features for all linears in this layer
        input_feat, layer_output = _collect_layer_input_feat(
            layer, calib_hidden, module_kwargs
        )

        # 2b. For each scaling pair: find best scales, absorb into weights
        scaling_pairs = _get_qwen2_scaling_pairs(layer, input_feat, module_kwargs)

        for pair in scaling_pairs:
            inp = pair.get("inp")
            if inp is None:
                continue

            inspect_module = pair.get("module2inspect", pair["linears"][0])
            pair_kwargs = pair.get("kwargs", {})

            best_scales = _search_best_scale(
                module2inspect=inspect_module,
                linears=pair["linears"],
                x_samples=inp.to(device),
                bits=bits,
                group_size=group_size,
                n_grid=n_grid,
                kwargs=pair_kwargs,
            )

            # Absorb scales into prev_op and linear weights
            _apply_awq_scale(
                pair["prev_op"],
                pair["linears"],
                best_scales.to(device),
            )

        # 2c. Quantize all Linear layers in this layer with RTN
        _quantize_layer_rtn(layer, bits=bits, group_size=group_size, skip_modules=skip_modules)

        # 2d. Forward calibration data through quantized layer → update for next layer
        with torch.no_grad():
            out = layer(calib_hidden, **module_kwargs)
            if isinstance(out, tuple):
                calib_hidden = out[0]
            else:
                calib_hidden = out

        print(f" ✓")

    # Step 3: Quantize LM head (unless skipped)
    if "lm_head" not in skip_modules:
        _quantize_single_linear(model, "lm_head", bits=bits, group_size=group_size)

    print(f"  AWQ: Done ({num_layers} layers quantized)")
    return model


def _quantize_layer_rtn(
    layer: nn.Module, bits: int = 4, group_size: int = 128,
    skip_modules: set[str] | None = None,
):
    """Quantize all Linear layers within a decoder layer using RTN."""
    if skip_modules is None:
        skip_modules = set()

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    replacements = {}
    for name, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in skip_modules:
            continue

        w = module.weight.data
        out_features, in_features = w.shape
        has_bias = module.bias is not None

        ql = QuantizedLinear(
            in_features, out_features, bias=has_bias,
            bits=bits, group_size=group_size,
        )

        if group_size > 0:
            num_groups = (in_features + group_size - 1) // group_size
            padded_in = num_groups * group_size
            w_padded = F.pad(w, (0, padded_in - in_features)) if padded_in > in_features else w
            w_groups = w_padded.reshape(out_features, num_groups, group_size)
            scale = w_groups.abs().amax(dim=2).clamp(min=1e-10) / qmax
            int_w = (w_groups / scale.unsqueeze(-1)).round().clamp(qmin, qmax).to(torch.int8)
            int_w = int_w.reshape(out_features, padded_in)
            if padded_in > in_features:
                int_w = int_w[:, :in_features]
            ql.int_weight.copy_(int_w)
            ql.scales.copy_(scale.to(torch.float16))
        else:
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
            int_w = (w / scale).round().clamp(qmin, qmax).to(torch.int8)
            ql.int_weight.copy_(int_w)
            ql.scales.copy_(scale.to(torch.float16))

        if has_bias:
            ql.bias.copy_(module.bias.data.to(torch.float16))

        replacements[name] = ql

    for name, ql in replacements.items():
        parts = name.split(".")
        parent = layer
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], ql.to(next(layer.parameters()).device))


def _quantize_single_linear(
    model: nn.Module, attr_name: str, bits: int = 4, group_size: int = 128,
):
    """Quantize a single top-level Linear layer (e.g., lm_head)."""
    module = getattr(model, attr_name, None)
    if module is None or not isinstance(module, nn.Linear):
        return

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    w = module.weight.data
    out_features, in_features = w.shape
    has_bias = module.bias is not None

    ql = QuantizedLinear(
        in_features, out_features, bias=has_bias,
        bits=bits, group_size=group_size,
    )

    if group_size > 0:
        num_groups = (in_features + group_size - 1) // group_size
        padded_in = num_groups * group_size
        w_padded = F.pad(w, (0, padded_in - in_features)) if padded_in > in_features else w
        w_groups = w_padded.reshape(out_features, num_groups, group_size)
        scale = w_groups.abs().amax(dim=2).clamp(min=1e-10) / qmax
        int_w = (w_groups / scale.unsqueeze(-1)).round().clamp(qmin, qmax).to(torch.int8)
        int_w = int_w.reshape(out_features, padded_in)
        if padded_in > in_features:
            int_w = int_w[:, :in_features]
        ql.int_weight.copy_(int_w)
        ql.scales.copy_(scale.to(torch.float16))
    else:
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
        int_w = (w / scale).round().clamp(qmin, qmax).to(torch.int8)
        ql.int_weight.copy_(int_w)
        ql.scales.copy_(scale.to(torch.float16))

    if has_bias:
        ql.bias.copy_(module.bias.data.to(torch.float16))

    setattr(model, attr_name, ql.to(next(model.parameters()).device))
