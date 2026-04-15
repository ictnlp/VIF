import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)


# =============================================================================
# Utility Functions
# =============================================================================

def maybe_zero_3(param, ignore_status=False, name=None):
    """Handle DeepSpeed ZeRO-3 parameters."""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.warning(f"Parameter {name} is not available in ZeRO-3")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """Extract MM adapter parameters with ZeRO-3 compatibility."""
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """Split indices into roughly even chunks."""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """Group samples by modality and sequence length."""
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    # Check whether all samples belong to the same modality.
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    
    # Split multimodal and language-only samples.
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # Shuffle both groups independently.
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    
    # Assemble megabatches.
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # Merge the final incomplete batch.
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    
    # Shuffle megabatches.
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """Group samples only by length."""
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# =============================================================================
# Custom Sampler
# =============================================================================

class LengthGroupedSampler(Sampler):
    """Sampler that groups similar lengths to reduce padding overhead."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths 
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


# =============================================================================
# Main Trainer
# =============================================================================

class LLaVATrainer(Trainer):
    """
    LLaVA Trainer with Attention Bias Module Support
    
    Main features:
    1. Multimodal length-grouped sampling.
    2. Separate learning rate for the MM projector.
    3. Logging support for attention-bias losses and metrics.
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Build the training sampler."""
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Create the optimizer and optionally use a separate MM-projector LR.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # Collect parameters that should use weight decay.
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Use a separate learning rate for the MM projector when requested.
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            # Handle 8-bit Adam.
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"Skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"Total skipped: {skipped/2**20}M params")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss.
        
        Total Loss = LM Loss + KL Loss + Sparsity Loss
        """
        outputs = model(**inputs)
        
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # The model already returns the combined loss.
        loss = outputs["loss"]
        
        # Keep outputs for logging on the main process only.
        if self.args.local_rank in [-1, 0]:
            self._current_outputs = outputs
        
        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        """
        Handle logging, saving, and evaluation.
        """
        grad_norm = kwargs.pop("grad_norm", None)
        model = kwargs.pop("model", None)
        trial = kwargs.pop("trial", None)
        epoch = kwargs.pop("epoch", None)
        ignore_keys_for_eval = kwargs.pop("ignore_keys_for_eval", None)

        # transformers>=4.40: (tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
        # older versions:      (tr_loss, model, trial, epoch, ignore_keys_for_eval)
        if len(args) == 5:
            grad_norm, model, trial, epoch, ignore_keys_for_eval = args
        elif len(args) == 4:
            model, trial, epoch, ignore_keys_for_eval = args
        elif len(args) != 0:
            raise TypeError(
                f"Unexpected _maybe_log_save_evaluate args length: {len(args)}"
            )

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}

            # 1. Standard metrics
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()

            # 2. Extract custom metrics from model outputs.
            if hasattr(self, '_current_outputs') and self._current_outputs is not None:
                outputs = self._current_outputs
                
                if isinstance(outputs, dict):
                    # Map metric keys to their display names.
                    metrics_to_log = {
                        "text_loss": "text_loss",
                        "kl_loss": "kl_loss",
                        "sparsity_loss": "sparsity_loss",
                        "mix_entropy": "mix_entropy",
                    }
                    
                    for key, display_name in metrics_to_log.items():
                        if key in outputs:
                            val = outputs[key]
                            if torch.is_tensor(val):
                                logs[display_name] = round(val.item(), 4)
                            else:
                                logs[display_name] = round(float(val), 4)
                    
                    # Clear cached outputs to avoid memory growth.
                    self._current_outputs = None

            # 3. Update the aggregated loss.
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            # 4. Emit logs.
            self.log(logs)

        # 5. Evaluate and save.
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # LR scheduler
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save a checkpoint.

        - If tune_mm_mlp_adapter=True, save only the MM projector.
        - Otherwise, save the full model.
        """
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Save only the adapter weights.
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
                logger.info(f"✅ Saved MM Projector to {output_dir}")
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
            logger.info(f"✅ Saved full model checkpoint at step {self.state.global_step}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save the model."""
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass  # No extra action is required when only adapters are saved.
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
