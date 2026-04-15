"""
Selective layer patching for a chosen subset of transformer blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Set
import transformers.models.llama.modeling_llama as llama_modeling

_CURRENT_ATTN_BIAS = None
_ATTN_BIAS_ALPHA: Optional[float] = None
_ACTIVE_LAYERS: Set[int] = set()
_LAYER_COUNTER = {}
_ORIGINAL_FORWARDS = {}  # Preserve original forwards for restoration.


def _build_causal_mask(
    q_len: int,
    kv_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.full((q_len, kv_seq_len), torch.finfo(dtype).min, device=device)
    mask = torch.triu(mask, diagonal=1 + kv_seq_len - q_len)
    return mask.unsqueeze(0).unsqueeze(0)  # [1,1,Q,K]


def _prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    bsz: int,
    q_len: int,
    kv_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return _build_causal_mask(q_len, kv_seq_len, dtype, device).expand(bsz, 1, q_len, kv_seq_len)

    if attention_mask.dim() == 2:
        causal = _build_causal_mask(q_len, kv_seq_len, dtype, device).expand(bsz, 1, q_len, kv_seq_len)
        padding_mask = attention_mask == 0
        if padding_mask.any():
            padding_mask = padding_mask[:, None, None, :].expand(bsz, 1, q_len, kv_seq_len)
            padding_mask = padding_mask.to(dtype=dtype) * torch.finfo(dtype).min
            return causal + padding_mask
        return causal

    if attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)

    if attention_mask.dtype == torch.bool:
        attn_mask = torch.zeros_like(attention_mask, dtype=dtype, device=device)
        attn_mask = attn_mask.masked_fill(~attention_mask, torch.finfo(dtype).min)
        return attn_mask

    return attention_mask.to(dtype=dtype)


def set_attention_bias(
    bias: Optional[torch.Tensor],
    active_layers: Optional[Set[int]] = None,
    alpha: Optional[float] = None,
):
    global _CURRENT_ATTN_BIAS, _ACTIVE_LAYERS, _ATTN_BIAS_ALPHA
    _CURRENT_ATTN_BIAS = bias
    _ACTIVE_LAYERS = active_layers if active_layers is not None else set()
    if alpha is not None:
        _ATTN_BIAS_ALPHA = float(alpha)


def get_attention_bias() -> Optional[torch.Tensor]:
    global _CURRENT_ATTN_BIAS
    return _CURRENT_ATTN_BIAS


def reset_layer_counter():
    global _LAYER_COUNTER
    _LAYER_COUNTER.clear()


def get_current_layer_index(module_id: int) -> int:
    global _LAYER_COUNTER
    if module_id not in _LAYER_COUNTER:
        _LAYER_COUNTER[module_id] = len(_LAYER_COUNTER)
    return _LAYER_COUNTER[module_id]


def patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Patched forward - only for selected layers.
    Key fixes:
      1) Apply attention_mask on logits (before softmax), NOT on probs
      2) Mix vision attn_bias on probs only for rows that have bias (question rows)
      3) After mixing, re-apply hard visibility mask and renormalize
    """
    bsz, q_len, _ = hidden_states.size()
    incoming_past_key_value = past_key_value

    # Get the actual layer index.
    if hasattr(self, "layer_idx"):
        layer_idx = self.layer_idx
    else:
        layer_idx = get_current_layer_index(id(self))

    # Q, K, V projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # kv_seq_len
    if past_key_value is not None:
        if cache_position is not None:
            kv_seq_len = int(cache_position[-1].item()) + 1
        elif hasattr(past_key_value, "get_seq_length"):
            kv_seq_len = past_key_value.get_seq_length() + q_len
        else:
            kv_seq_len = past_key_value[0].shape[-2] + q_len
    else:
        kv_seq_len = q_len

    # RoPE
    if position_ids is None:
        if cache_position is not None:
            position_ids = cache_position
        else:
            start = kv_seq_len - q_len
            position_ids = torch.arange(start, kv_seq_len, device=hidden_states.device)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.size(0) == 1 and bsz > 1:
            position_ids = position_ids.expand(bsz, -1)

    try:
        # transformers>=4.40
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = llama_modeling.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
    except TypeError:
        # Backward compatibility for older rotary embedding signatures
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = llama_modeling.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

    # KV cache
    if past_key_value is not None:
        if hasattr(past_key_value, "update"):
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, layer_idx, cache_kwargs
            )
        else:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None
    else:
        past_key_value = (key_states, value_states) if use_cache else None

    # Repeat KV for GQA
    key_states = llama_modeling.repeat_kv(key_states, self.num_key_value_groups)
    value_states = llama_modeling.repeat_kv(value_states, self.num_key_value_groups)

    attn_bias = get_attention_bias()
    is_prefill = (q_len > 1)
    use_bias = attn_bias is not None and is_prefill and attn_bias.size(-1) == kv_seq_len

    if not use_bias:
        original = _ORIGINAL_FORWARDS.get(id(self))
        if original is not None:
            return original(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=incoming_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

    attention_mask = _prepare_attention_mask(
        attention_mask,
        bsz,
        q_len,
        kv_seq_len,
        query_states.dtype,
        query_states.device,
    )

    # -----------------------------
    # 1) Attention logits
    # -----------------------------
    attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
    # attn_scores: [B, H, Q, K]

    # -----------------------------
    # 2) Apply additive attention_mask on logits (BEFORE softmax)
    # attention_mask expected: [B, 1, Q, K] (visible=0, invisible=-inf/neg)
    # -----------------------------
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask.to(attn_scores.dtype)

    # -----------------------------
    # 3) Softmax -> probs
    # use float32 softmax for stability
    # -----------------------------
    attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # -----------------------------
    # 4) Optional vision bias mixing (probs space), only in prefill
    # -----------------------------
    if use_bias and attn_bias.size(-1) == attn_probs.size(-1):
        # Make attn_bias broadcastable to [B,H,Q,K]
        if attn_bias.dim() == 4:
            # [B,1,Q,K] or [B,H,Q,K]
            pass
        elif attn_bias.dim() == 3:
            # [B,Q,K] -> [B,1,Q,K]
            attn_bias = attn_bias.unsqueeze(1)
        elif attn_bias.dim() == 2:
            # [Q,K] -> [1,1,Q,K]
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported attn_bias shape: {attn_bias.shape}")

        # Expand heads if needed
        if attn_bias.size(1) == 1 and attn_probs.size(1) != 1:
            attn_bias = attn_bias.expand(-1, attn_probs.size(1), -1, -1)  # [B,H,Q,K]

        # Only mix rows (queries) that actually have bias mass (question rows)
        bias_rows = (attn_bias.sum(dim=-1, keepdim=True) > 0)  # [B,H,Q,1] bool

        alpha = _ATTN_BIAS_ALPHA if _ATTN_BIAS_ALPHA is not None else 0.5
        mixed = (1 - alpha) * attn_probs + alpha * attn_bias.to(attn_probs.dtype)
        attn_probs = torch.where(bias_rows, mixed, attn_probs)
        # Re-apply hard visibility constraints and renormalize
        if attention_mask is not None:
            # print(f"min max value of attention_mask: {attention_mask.min().item()}, {attention_mask.max().item()}")
            visible = (attention_mask == 0)  # [B,1,Q,K] bool
            if visible.size(1) == 1 and attn_probs.size(1) != 1:
                visible = visible.expand(-1, attn_probs.size(1), -1, -1)  # [B,H,Q,K]

            attn_probs = attn_probs.masked_fill(~visible, 0.0)
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-9)
    # -----------------------------
    # 5) Attention output
    # -----------------------------
    attn_probs = attn_probs.to(dtype=value_states.dtype)
    attn_output = torch.matmul(attn_probs, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_probs = None

    return attn_output, attn_probs, past_key_value


def apply_attention_patch_selective(model, active_layers: Set[int]):
    """
    Replace the self-attention forward only on selected layers.

    Args:
        model: LlavaLlamaModel instance.
        active_layers: Layer indices that should apply the bias.
    """
    global _ACTIVE_LAYERS, _ORIGINAL_FORWARDS
    _ACTIVE_LAYERS = active_layers
    
    patched_count = 0
    
    # LlavaLlamaModel inherits from LlamaModel, so layers live on self.layers.
    layers = model.layers if hasattr(model, 'layers') else model.model.layers
    
    # Iterate over all transformer layers.
    for layer_idx, layer in enumerate(layers):
        if layer_idx in active_layers:
            # Replace only the selected layer.
            module = layer.self_attn
            module_id = id(module)
            
            # Save the original forward method.
            if module_id not in _ORIGINAL_FORWARDS:
                _ORIGINAL_FORWARDS[module_id] = module.forward
            
            # Swap in the patched forward.
            import types
            module.forward = types.MethodType(patched_llama_attention_forward, module)
            
            patched_count += 1
            
            # if patched_count == 1:
            #     print(f"   First patched layer: {layer_idx} (self_attn)")
    
    # print(f"✅ Patched {patched_count} attention modules at layers {sorted(active_layers)}!")
    return patched_count > 0


def remove_attention_patch_selective(model):
    """Restore the original forward method on patched layers."""
    global _ORIGINAL_FORWARDS
    
    restored_count = 0
    
    # Handle layer access in the same way as the patch helper.
    layers = model.layers if hasattr(model, 'layers') else model.model.layers
    
    for layer in layers:
        module = layer.self_attn
        module_id = id(module)
        if module_id in _ORIGINAL_FORWARDS:
            module.forward = _ORIGINAL_FORWARDS[module_id]
            restored_count += 1
    
    _ORIGINAL_FORWARDS.clear()
    print(f"✅ Restored {restored_count} attention modules!")


# Keep the legacy API for compatibility.
def apply_attention_patch():
    """
    Deprecated: replace all attention layers globally.

    Prefer apply_attention_patch_selective().
    """
    print("⚠️ Warning: apply_attention_patch() replaces ALL layers.")
    print("   Recommended: use apply_attention_patch_selective() instead.")
    llama_modeling.LlamaAttention.forward = patched_llama_attention_forward
    print("✅ Attention intervention patch applied (ALL layers)!")
