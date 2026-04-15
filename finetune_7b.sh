#!/bin/bash

# Keep Python/BLAS thread counts bounded.
export OMP_NUM_THREADS=4             # Per-process OpenMP threads (4 * 4 procs = 16 total)
export MKL_NUM_THREADS=1             # Keep MKL single-threaded to avoid OMP conflicts
export OPENBLAS_NUM_THREADS=1        # Keep OpenBLAS single-threaded to avoid OMP conflicts
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

# CUDA/NCCL settings for performance and stability.
export CUDA_DEVICE_MAX_CONNECTIONS=1    # Keep NCCL connections stable without hurting throughput
export NCCL_ASYNC_ERROR_HANDLING=1      # Catch async errors and avoid rank deadlocks
export NCCL_BLOCKING_WAIT=1             # Surface failures immediately
export NCCL_DEBUG=WARN                  # NCCL log level
export TORCH_DISTRIBUTED_DEBUG=OFF      # PyTorch distributed log level
export NCCL_BUFFSIZE=67108864           # 64 MB NCCL communication buffer
export NCCL_IB_DISABLE=0                # Enable InfiniBand when available
export NCCL_NET_GDR_LEVEL=2     # GPU Direct RDMA
export NCCL_TIMEOUT=3600               # 1 hour NCCL timeout
export NCCL_P2P_DISABLE=0              # Enable P2P communication

export BATCH_SIZE=8
export GRADIENT_ACCU_STEPS=16

export DATA_PATH=/path/to/datasets/llava_v1_5_mix665k.json
export SAVE_PATH=llava-vif_7b_bs512_epoch1_gmm16
export BASE_LR=2e-5
export VIT_LR=2e-6
export LATENT_LEARN_START=11
export LATENT_LEARN_END=17
export LATENT_APPLY_START=25
export LATENT_APPLY_END=31
export LATENT_LAYER_STRIDE=2
export LATENT_KL_WEIGHT=0.1
export LATENT_SPARSITY_WEIGHT=0.1
export LATENT_ENTROPY_WEIGHT=1.0
export LATENT_VOLUME_WEIGHT=1.0
export LATENT_NUM_COMPONENTS=16
export LATENT_MIN_SIGMA=0.035
export LATENT_BIAS_ALPHA=0.5

# --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
# export SAVE_PATH=llava-lora-vif_7b_bs512_epoch1_gmm16

deepspeed --include localhost:0,1,2,3 \
    train_mem.py \
    --mm_projector_lr ${BASE_LR} \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /path/to/llava-v1.5-7b/ \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder /path/to/datasets/ \
    --vision_tower /path/to/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --tf32 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --accelerator_config '{"gradient_accumulation_kwargs":{"sync_each_batch":true}}' \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_latent_importance True \
    --latent_learning_start ${LATENT_LEARN_START} \
    --latent_learning_end ${LATENT_LEARN_END} \
    --latent_apply_start ${LATENT_APPLY_START} \
    --latent_apply_end ${LATENT_APPLY_END} \
    --latent_layer_stride ${LATENT_LAYER_STRIDE} \
    --latent_num_layers 3 \
    --latent_kl_weight ${LATENT_KL_WEIGHT} \
    --latent_sparsity_weight ${LATENT_SPARSITY_WEIGHT} \
    --latent_entropy_weight ${LATENT_ENTROPY_WEIGHT} \
    --latent_volume_weight ${LATENT_VOLUME_WEIGHT} \
    --latent_num_components ${LATENT_NUM_COMPONENTS} \
    --latent_min_sigma ${LATENT_MIN_SIGMA} \
    --latent_bias_alpha ${LATENT_BIAS_ALPHA}
