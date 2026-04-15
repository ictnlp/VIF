from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..latent_importance import AttentionBiasGenerator
from ..attention_intervention import (
    apply_attention_patch, 
    apply_attention_patch_selective,
    set_attention_bias, 
    reset_layer_counter
)


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

    def __init__(self, **kwargs):
        self.use_latent_importance = kwargs.pop("use_latent_importance", False)
        self.latent_hidden_size = kwargs.pop("latent_hidden_size", None)
        self.latent_num_layers = kwargs.pop("latent_num_layers", 2)
        self.latent_num_heads = kwargs.pop("latent_num_heads", 4)
        self.latent_ffn_dim = kwargs.pop("latent_ffn_dim", None)
        self.latent_dropout = kwargs.pop("latent_dropout", 0.1)
        self.latent_layer_stride = kwargs.pop("latent_layer_stride", 2)
        self.latent_kl_weight = kwargs.pop("latent_kl_weight", 0.1)
        self.latent_sparsity_weight = kwargs.pop("latent_sparsity_weight", 0.1)
        self.latent_entropy_weight = kwargs.pop("latent_entropy_weight", 1.0)
        self.latent_volume_weight = kwargs.pop("latent_volume_weight", 1.0)
        self.latent_num_components = kwargs.pop("latent_num_components", 16)
        self.latent_min_sigma = kwargs.pop("latent_min_sigma", 0.035)
        self.latent_bias_alpha = kwargs.pop("latent_bias_alpha", 0.5)

        # Layer-control aliases kept for backward compatibility with older scripts.
        latent_l_start = kwargs.pop("latent_l_start", None)
        latent_l_end = kwargs.pop("latent_l_end", None)
        latent_start_layer = kwargs.pop("latent_start_layer", None)
        latent_end_layer = kwargs.pop("latent_end_layer", None)

        self.latent_l_start = latent_l_start
        self.latent_l_end = latent_l_end
        self.latent_start_layer = latent_start_layer
        self.latent_end_layer = latent_end_layer

        self.latent_learning_start = kwargs.pop("latent_learning_start", latent_l_start)
        self.latent_learning_end = kwargs.pop("latent_learning_end", latent_l_end)
        self.latent_apply_start = kwargs.pop("latent_apply_start", latent_start_layer)
        self.latent_apply_end = kwargs.pop("latent_apply_end", latent_end_layer)
        
        super().__init__(**kwargs)

        if self.latent_hidden_size is None:
            self.latent_hidden_size = self.hidden_size
        if self.latent_ffn_dim is None:
            self.latent_ffn_dim = self.latent_hidden_size * 4


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        
        self._latent_context: Optional[Dict[str, torch.Tensor]] = None
        self._latent_metrics: Optional[Dict[str, torch.Tensor]] = None
        self._bias_cache: Dict[int, torch.Tensor] = {}

        if getattr(config, "use_latent_importance", False):
            print("✅ Applying selective attention intervention patch...")
            num_layers = len(self.layers)
            stride = getattr(config, "latent_layer_stride", 2) or 1

            def build_layers(start, end):
                if start is None or end is None:
                    return []
                start = max(0, min(start, num_layers - 1))
                end = max(0, min(end, num_layers - 1))
                if start > end:
                    start, end = end, start
                return list(range(start, end + 1, stride))

            def default_learning_range():
                start = int(num_layers * 0.34)
                end = int(num_layers * 0.5)
                return max(0, start), max(start, end)

            def default_apply_range():
                start = int(num_layers * 0.75)
                end = num_layers - 1
                return max(0, start), max(start, end)

            learning_start = config.latent_learning_start
            learning_end = config.latent_learning_end
            if learning_start is None or learning_end is None:
                if config.latent_l_start is not None and config.latent_l_end is not None:
                    learning_start, learning_end = config.latent_l_start, config.latent_l_end
                else:
                    learning_start, learning_end = default_learning_range()

            apply_start = config.latent_apply_start
            apply_end = config.latent_apply_end
            if apply_start is None or apply_end is None:
                if config.latent_start_layer is not None and config.latent_end_layer is not None:
                    apply_start, apply_end = config.latent_start_layer, config.latent_end_layer
                else:
                    apply_start, apply_end = default_apply_range()

            self.learning_layers = build_layers(learning_start, learning_end)
            self.apply_layers = build_layers(apply_start, apply_end)

            if not self.learning_layers:
                raise ValueError("No learning layers configured for latent importance.")

            if not self.apply_layers or len(self.learning_layers) != len(self.apply_layers):
                candidates = list(range(0, num_layers, stride))
                if len(candidates) >= len(self.learning_layers):
                    self.apply_layers = candidates[-len(self.learning_layers):]
                else:
                    min_len = min(len(self.learning_layers), len(candidates))
                    self.learning_layers = self.learning_layers[:min_len]
                    self.apply_layers = candidates

            self.layer_mapping = dict(zip(self.learning_layers, self.apply_layers))
            
            # Patch only the layers that receive the injected bias.
            apply_attention_patch_selective(self, set(self.apply_layers))
            
            print("✅ Initializing AttentionBiasGenerator...")
            self.attention_bias_generator = AttentionBiasGenerator(config)

            self._register_bias_hooks()
            
            print(f"📍 Layer mapping:")
            for learn, apply in self.layer_mapping.items():
                print(f"   Layer {learn} → Layer {apply}")
        else:
            self.attention_bias_generator = None
            self.learning_layers = []
            self.apply_layers = []
            self.layer_mapping = {}

    def _register_bias_hooks(self):
        """Generate the bias right after each learning layer without storing features."""
        self._hooks = []
        
        for learn_layer_idx in self.learning_layers:
            apply_layer_idx = self.layer_mapping[learn_layer_idx]
            
            # Step 1: generate the bias right after the learning layer runs.
            def make_generation_hook(learn_idx, apply_idx):
                def hook_fn(module, input, output):
                    # Compute only when latent context exists and the target bias is missing.
                    if (
                        self._latent_context is not None
                        and apply_idx not in self._bias_cache
                    ):
                        # Use the current layer output directly rather than caching features.
                        hidden_states = output[0]
                        
                        # Generate the attention bias immediately.
                        attn_bias, metrics = self.attention_bias_generator(
                            inputs_embeds=hidden_states,
                            question_mask=self._latent_context["question_mask"],
                            answer_mask=self._latent_context["answer_mask"],
                            vision_mask=self._latent_context["vision_mask"],
                            training=self.training,
                        )
                        
                        # Keep gradients in training, detach in inference to save memory.
                        if self.training:
                            self._bias_cache[apply_idx] = attn_bias
                        else:
                            self._bias_cache[apply_idx] = attn_bias.detach()
                        
                        # Accumulate metrics once per forward pass.
                        if self._latent_metrics is None:
                            self._latent_metrics = metrics
                
                return hook_fn
            
            # Register the generation hook on the learning layer.
            generation_hook = self.layers[learn_layer_idx].register_forward_hook(
                make_generation_hook(learn_layer_idx, apply_layer_idx)
            )
            self._hooks.append(generation_hook)
            
            # Step 2: inject the cached bias before the apply layer runs.
            def make_injection_hook(apply_idx):
                def pre_hook_fn(module, input):
                    # Pull the corresponding bias from the cache.
                    if apply_idx in self._bias_cache:
                        attn_bias = self._bias_cache[apply_idx]
                        # Activate the bias only for this layer.
                        set_attention_bias(
                            attn_bias,
                            active_layers={apply_idx},
                            alpha=getattr(self.config, "latent_bias_alpha", None),
                        )
                
                return pre_hook_fn
            
            injection_hook = self.layers[apply_layer_idx].register_forward_pre_hook(
                make_injection_hook(apply_layer_idx)
            )
            self._hooks.append(injection_hook)

    def set_latent_context(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        question_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        vision_mask: torch.Tensor,
        labels_present: bool,
        vision_token_spans: Optional[torch.Tensor] = None,
        cls_features: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Store the latent-importance context for the current forward pass."""
        if not getattr(self.config, "use_latent_importance", False):
            return
        
        self._latent_context = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "question_mask": question_mask,
            "answer_mask": answer_mask,
            "vision_mask": vision_mask,
            "labels_present": labels_present,
            "vision_token_spans": vision_token_spans,
            "cls_features": cls_features,
        }

    def consume_latent_metrics(self) -> Dict[str, torch.Tensor]:
        """Return latent metrics and clear the stored copy."""
        metrics = self._latent_metrics
        self._latent_metrics = None
        return metrics if metrics is not None else {}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Rely on hooks to generate and inject attention bias automatically."""
        reset_layer_counter()
        
        # Recompute every full forward; keep the cache across decode steps.
        if self.training or past_key_values is None:
            self._bias_cache.clear()
        
        try:
            # Hooks handle all bias generation and injection work.
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        finally:
            # Clear the active bias after the forward pass.
            set_attention_bias(None, active_layers=None)

        return outputs

    def __del__(self):
        """Remove registered hooks during cleanup."""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Add regularization terms only while training.
        latent_metrics = self.model.consume_latent_metrics()
        if latent_metrics and outputs.loss is not None and self.training:
            kl_loss = latent_metrics.get("kl_div", torch.tensor(0.0, device=outputs.loss.device))
            sparsity_loss = latent_metrics.get("sparsity_loss", torch.tensor(0.0, device=outputs.loss.device))

            # Expose auxiliary values for logging.
            outputs["text_loss"] = outputs.loss.detach()
            outputs["kl_loss"] = kl_loss.detach()
            outputs["sparsity_loss"] = sparsity_loss.detach()
            outputs["mix_entropy"] = latent_metrics.get("mix_entropy", torch.tensor(0.0)).detach()
            
            # Total loss = LM loss + auxiliary regularizers.
            outputs.loss = (
                outputs.loss +
                self.config.latent_kl_weight * kl_loss +
                self.config.latent_sparsity_weight * sparsity_loss

            )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
