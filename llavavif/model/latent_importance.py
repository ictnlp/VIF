# Copyright 2026
#
# Variational Latent Importance Module for Vision Tokens
# Based on Variational Inference and Bayesian Learning
#
# Theory:
# - Prior p(z|I,Q): predicts importance without answer (inference time)
# - Posterior q(z|I,Q,A): predicts importance with answer (training time)
# - ELBO: E_q[log p(A|I,Q,z)] - KL(q(z|I,Q,A) || p(z|I,Q))
#
# Usage:
#   Training: Sample from posterior q(z|I,Q,A), minimize KL(q||p)
#   Inference: Sample from prior p(z|I,Q) (no answer available)

"""
Attention Bias Generator with Enhanced Cross-Modal Interaction
Core ideas:
1. Bidirectional interaction between vision and text.
2. Token-level alignment between text tokens and image regions.
3. Answer-guided posterior learning during training.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class SpatialGaussianRenderer(nn.Module):
    """Decode latent slots into a spatial Gaussian distribution."""
    def __init__(self, latent_dim, num_patches_side=24, min_sigma: float = 0.035):
        super().__init__()
        self.H = num_patches_side
        self.W = num_patches_side
        self.min_sigma = min_sigma
        
        # Predict Gaussian parameters.
        self.pos_head = nn.Linear(latent_dim, 2)
        self.sigma_head = nn.Linear(latent_dim, 1)
        self.weight_head = nn.Linear(latent_dim, 1)
        
        # Initialize positions broadly and sigma conservatively.
        nn.init.normal_(self.pos_head.weight, std=0.01)
        nn.init.uniform_(self.pos_head.bias, -1.0, 1.0)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
        nn.init.constant_(self.sigma_head.bias, -2.0)
        nn.init.normal_(self.weight_head.weight, std=0.01)
        nn.init.constant_(self.weight_head.bias, 0.0)
        
        self.grid = None

    def _get_grid(self, device):
        if self.grid is None or self.grid.device != device:
            y, x = torch.meshgrid(
                torch.linspace(0, 1, self.H, device=device), 
                torch.linspace(0, 1, self.W, device=device), 
                indexing='ij'
            )
            self.grid = torch.stack([x, y], dim=-1).reshape(1, -1, 2)
        return self.grid

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size, num_slots, _ = z.shape
        device = z.device
        
        mu_pos = torch.sigmoid(self.pos_head(z))  # [B, K, 2] center coordinates
        sigma = F.softplus(self.sigma_head(z)) + self.min_sigma  # [B, K, 1] standard deviation
        mix_logits = self.weight_head(z).squeeze(-1)  # [B, K]
        mix_weights = F.softmax(mix_logits, dim=-1)
        
        grid = self._get_grid(device)  # [1, N, 2]
        grid_expanded = grid.unsqueeze(1)  # [1, 1, N, 2] broadcast to [B, 1, N, 2]
        mu_expanded = mu_pos.unsqueeze(2)  # [B, K, 1, 2]
        
        dist_sq = torch.sum((grid_expanded - mu_expanded) ** 2, dim=-1)
        gaussian_blobs = torch.exp(-dist_sq / (2 * sigma.squeeze(-1).unsqueeze(-1) ** 2))
        weighted_blobs = gaussian_blobs * mix_weights.unsqueeze(-1)
        heatmap_raw = weighted_blobs.sum(dim=1)  # [B, N]
        # heatmap = heatmap_raw / (heatmap_raw.sum(dim=-1, keepdim=True) + 1e-6)
        heatmap = F.softmax(heatmap_raw, dim=-1)
        
        return heatmap, (mu_pos, sigma, mix_weights)


class BidirectionalCrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention block.

    1. Vision queries text for semantics.
    2. Text queries vision for visual evidence.
    3. Both modalities observe one another before fusion.
    """
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        
        # Vision Self-Attention
        self.vision_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.vision_norm1 = nn.LayerNorm(hidden_dim)
        
        # Text Self-Attention
        self.text_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_norm1 = nn.LayerNorm(hidden_dim)
        
        # Vision -> Text cross-attention.
        self.v2t_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.vision_norm2 = nn.LayerNorm(hidden_dim)
        
        # Text -> Vision cross-attention.
        self.t2v_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN for Vision
        self.vision_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.vision_norm3 = nn.LayerNorm(hidden_dim)
        
        # FFN for Text
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.text_norm3 = nn.LayerNorm(hidden_dim)
        
        # self.dropout = nn.Dropout(dropout)

    def forward(self, vision, text, text_mask=None):
        """
        Args:
            vision: [batch, n_vision, dim]
            text: [batch, n_text, dim]
            text_mask: [batch, n_text] (True = valid token)
        
        Returns:
            vision: [batch, n_vision, dim] enhanced vision features
            text: [batch, n_text, dim] enhanced text features
        """
        # 1. Self-Attention on Vision
        v = self.vision_norm1(vision)
        v, _ = self.vision_self_attn(v, v, v)
        vision = vision + v
        
        if text is not None:
            # 2. Self-Attention on Text
            t = self.text_norm1(text)
            t, _ = self.text_self_attn(t, t, t)
            text = text + t
            
            # 3. Vision queries text for semantics.
            v = self.vision_norm2(vision)
            padding_mask = ~text_mask if text_mask is not None else None
            v, _ = self.v2t_attn(query=v, key=text, value=text, key_padding_mask=padding_mask)
            vision = vision + v
            
            # 4. Text queries vision for visual evidence.
            t = self.text_norm2(text)
            t, _ = self.t2v_attn(query=t, key=vision, value=vision)
            text = text + t
        
        # 5. FFN for Vision
        v = self.vision_norm3(vision)
        vision = vision + self.vision_ffn(v)
        
        # 6. FFN for Text
        if text is not None:
            t = self.text_norm3(text)
            text = text + self.text_ffn(t)
        
        return vision, text


class EnhancedCrossModalEncoder(nn.Module):
    """Bidirectional vision-text encoder returning both feature streams."""
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_size
        latent_dim = getattr(config, "latent_hidden_size", hidden_dim)
        num_layers = getattr(config, "latent_num_layers", 2) or 2
        num_heads = getattr(config, "latent_num_heads", 8) or 8
        ffn_dim = getattr(config, "latent_ffn_dim", None) or latent_dim * 2
        # dropout = 0.1
        
        self.img_proj = nn.Linear(hidden_dim, latent_dim)
        self.txt_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.layers = nn.ModuleList([
            BidirectionalCrossAttentionLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        
        self.vision_norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(latent_dim)

    def forward(self, vision_tokens, text_tokens, text_mask):
        """
        Returns:
            vision_feat: [batch, n_vision, latent_dim]
            text_feat: [batch, n_text, latent_dim] or None
        """
        vision = self.img_proj(vision_tokens)
        text = self.txt_proj(text_tokens) if text_tokens is not None else None
        
        for layer in self.layers:
            vision, text = layer(vision, text, text_mask)
        
        vision = self.vision_norm(vision)
        if text is not None:
            text = self.text_norm(text)
        
        return vision, text

class AttentionBiasGenerator(nn.Module):
    """
    Attention bias generator with:
    1. bidirectional feature interaction,
    2. optional token-level alignment,
    3. answer-guided importance learning.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        latent_dim = getattr(config, "latent_hidden_size", hidden_size)
        self.num_latent_slots = getattr(config, "latent_num_components", 16)
        self.entropy_weight = getattr(config, "latent_entropy_weight", 1.0)
        self.volume_weight = getattr(config, "latent_volume_weight", 1.0)
        
        # Bidirectional encoders.
        self.prior_encoder = EnhancedCrossModalEncoder(config)
        self.posterior_encoder = EnhancedCrossModalEncoder(config)
        
        # Compress vision features into latent slots.
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_latent_slots, latent_dim))
        compress_heads = getattr(config, "latent_num_heads", 8) or 8
        self.compress_attn = nn.MultiheadAttention(latent_dim, num_heads=compress_heads, batch_first=True)
        
        # VAE Heads
        self.prior_mean_head = nn.Linear(latent_dim, latent_dim)
        self.prior_logvar_head = nn.Linear(latent_dim, latent_dim)
        self.posterior_mean_head = nn.Linear(latent_dim, latent_dim)
        self.posterior_logvar_head = nn.Linear(latent_dim, latent_dim)
        
        # Renderer
        min_sigma = getattr(config, "latent_min_sigma", 0.035)
        self.renderer = SpatialGaussianRenderer(latent_dim, num_patches_side=24, min_sigma=min_sigma)
        
        self.inference_cache = []

    def clear_cache(self):
        """Clear cached inference artifacts."""
        self.inference_cache = []

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        question_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        vision_mask: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len, hidden_dim = inputs_embeds.size()
        device = inputs_embeds.device
        
        # --- Extract token groups ---
        vision_counts = vision_mask.sum(dim=1)
        if torch.all(vision_counts == 0):
            return torch.zeros(batch_size, seq_len, seq_len, device=device), {}
        
        n_vision = vision_counts[0].item()
        vision_tokens = self._extract_fixed_length_tokens_batch(inputs_embeds, vision_mask, n_vision)
        question_tokens, q_lengths = self._extract_variable_tokens_batch(inputs_embeds, question_mask)
        q_mask = torch.arange(q_lengths.max().item(), device=device).unsqueeze(0) < q_lengths.unsqueeze(1)
        
        use_posterior = training and answer_mask.any()
        
        # --- Encode prior branch ---
        prior_vision_feat, prior_text_feat = self.prior_encoder(vision_tokens, question_tokens, q_mask)
        
        # Compress to latent slots
        queries = self.latent_queries.expand(batch_size, -1, -1)
        prior_slot_feat, _ = self.compress_attn(queries, prior_vision_feat, prior_vision_feat)
        
        prior_mean = self.prior_mean_head(prior_slot_feat)
        prior_logvar = self.prior_logvar_head(prior_slot_feat).clamp(-5, 5)
        
        # --- Encode posterior branch with answer tokens ---
        if use_posterior:
            answer_tokens, a_lengths = self._extract_variable_tokens_batch(inputs_embeds, answer_mask)
            max_a = a_lengths.max().item()
            qa_tokens = torch.cat([question_tokens, answer_tokens], dim=1)
            a_mask = torch.arange(max_a, device=device).unsqueeze(0) < a_lengths.unsqueeze(1)
            qa_mask = torch.cat([q_mask, a_mask], dim=1)
            
            post_vision_feat, post_text_feat = self.posterior_encoder(vision_tokens, qa_tokens, qa_mask)
            post_slot_feat, _ = self.compress_attn(queries, post_vision_feat, post_vision_feat)
            
            post_mean = self.posterior_mean_head(post_slot_feat)
            post_logvar = self.posterior_logvar_head(post_slot_feat).clamp(-5, 5)
            
            # Sample z
            eps = torch.randn_like(post_mean)
            z = post_mean + torch.exp(0.5 * post_logvar) * eps
            
            # KL Loss
            kl_div = kl_divergence(
                Normal(post_mean, torch.exp(0.5 * post_logvar)),
                Normal(prior_mean, torch.exp(0.5 * prior_logvar))
            ).mean()
        else:
            eps = torch.randn_like(prior_mean)
            z = prior_mean + torch.exp(0.5 * prior_logvar) * eps
            kl_div = torch.zeros((), device=device)
        
        # --- Render heatmap ---
        importance_map, (pos, sigma, mix_weights) = self.renderer(z)
        gaussian_volume = 1.0 * sigma.squeeze(-1).pow(2).mean()
        
        # --- Generate attention bias ---
        text_mask = question_mask | answer_mask  # answer_mask is zero during inference
        attn_bias = self._generate_attention_bias(
            importance_map, vision_mask, text_mask, batch_size, seq_len, device
        )
        importance_entropy = -(importance_map * (importance_map + 1e-8).log()).sum(dim=-1).mean()
        sparsity_loss = self.entropy_weight * importance_entropy + self.volume_weight * gaussian_volume
        
        # --- Metrics ---
        metrics = {
            'kl_div': kl_div,
            'sparsity_loss': sparsity_loss,
            'mix_entropy': -(mix_weights * (mix_weights + 1e-8).log()).sum(dim=-1).mean(),
        }
        
        return attn_bias, metrics

    def _generate_attention_bias(
        self, importance_map, vision_mask, text_mask, batch_size, seq_len, device
    ):
        # importance_map: [B, 576], already normalized by softmax
        attn_bias = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)  # [B, 1, Q, K]

        for b in range(batch_size):
            vision_pos = vision_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [576]
            text_pos = text_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [n_text]

            if vision_pos.numel() == 0 or text_pos.numel() == 0:
                continue

            # The rendered heatmap must align with the visible vision-token count.
            if vision_pos.numel() != importance_map.size(-1):
                raise ValueError(
                    f"Mismatch: vision tokens={vision_pos.numel()} vs heatmap={importance_map.size(-1)}"
                )

            # Reuse the same region distribution for each selected text row.
            bias_row = importance_map[b]  # [576], sum=1

            # Scatter into the vision-token columns.
            attn_bias[b, 0, text_pos[:, None], vision_pos[None, :]] = bias_row[None, :]

        return attn_bias

    def _compute_diversity_loss(self, pos):
        batch_size, num_slots, _ = pos.shape
        
        pos1 = pos.unsqueeze(2)
        pos2 = pos.unsqueeze(1)
        dist = torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1) + 1e-6)
        
        mask = (1 - torch.eye(num_slots, device=pos.device)).unsqueeze(0)
        repulsion = F.relu(0.15 - dist) * mask
        repulsion_loss = repulsion.sum() / (batch_size * num_slots * (num_slots - 1))
        
        pos_std = pos.std(dim=1).mean()
        spread_loss = F.relu(0.25 - pos_std)
        
        return repulsion_loss + spread_loss

    def _extract_fixed_length_tokens_batch(self, inputs_embeds, mask, expected_count):
        batch_size, seq_len, hidden_dim = inputs_embeds.size()
        
        masked_tokens = inputs_embeds[mask]
        total_tokens = masked_tokens.size(0)
        expected_total = batch_size * expected_count

        if expected_total > 0 and total_tokens == expected_total:
            return masked_tokens.view(batch_size, expected_count, hidden_dim)

        print(f"Warning: Inconsistent number of vision tokens across batch. Using manual extraction. Expected total: {expected_total}, Actual total: {total_tokens}.")
        lengths = mask.sum(dim=1)  # [B]
        valid_lengths = lengths[lengths > 0]
        expected = int(valid_lengths.mode().values.item()) if valid_lengths.numel() > 0 else int(expected_count)

        out = torch.zeros(
            (batch_size, expected, hidden_dim),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        for b in range(batch_size):
            n = int(lengths[b].item())
            if n <= 0:
                continue  # Keep all-zero rows for samples without images.
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            tok = inputs_embeds[b, idx]  # [n, D]
            if n >= expected:
                out[b] = tok[:expected]
            else:
                out[b, :n] = tok

        return out

    def _extract_variable_tokens_batch(self, inputs_embeds, mask):
        batch_size, seq_len, hidden_dim = inputs_embeds.size()
        lengths = mask.sum(dim=1)
        max_len = lengths.max().item()
        
        if max_len == 0:
            return torch.zeros(batch_size, 0, hidden_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype), lengths
            
        tokens = torch.zeros(batch_size, max_len, hidden_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        for b in range(batch_size):
            n = lengths[b].item()
            if n > 0:
                indices = mask[b].nonzero(as_tuple=False).squeeze(-1)
                tokens[b, :n] = inputs_embeds[b, indices]
        return tokens, lengths
