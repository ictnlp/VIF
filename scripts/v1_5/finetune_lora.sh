#!/bin/bash

USE_LATENT_IMPORTANCE=${USE_LATENT_IMPORTANCE:-True}
LATENT_LEARN_START=${LATENT_LEARN_START:-10}
LATENT_LEARN_END=${LATENT_LEARN_END:-16}
LATENT_APPLY_START=${LATENT_APPLY_START:-24}
LATENT_APPLY_END=${LATENT_APPLY_END:-30}
LATENT_LAYER_STRIDE=${LATENT_LAYER_STRIDE:-2}
LATENT_KL_WEIGHT=${LATENT_KL_WEIGHT:-0.1}
LATENT_SPARSITY_WEIGHT=${LATENT_SPARSITY_WEIGHT:-0.1}
LATENT_ENTROPY_WEIGHT=${LATENT_ENTROPY_WEIGHT:-1.0}
LATENT_VOLUME_WEIGHT=${LATENT_VOLUME_WEIGHT:-1.0}
LATENT_NUM_COMPONENTS=${LATENT_NUM_COMPONENTS:-16}
LATENT_MIN_SIGMA=${LATENT_MIN_SIGMA:-0.035}
LATENT_BIAS_ALPHA=${LATENT_BIAS_ALPHA:-0.5}

deepspeed llavavif/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_latent_importance ${USE_LATENT_IMPORTANCE} \
    --latent_learning_start ${LATENT_LEARN_START} \
    --latent_learning_end ${LATENT_LEARN_END} \
    --latent_apply_start ${LATENT_APPLY_START} \
    --latent_apply_end ${LATENT_APPLY_END} \
    --latent_layer_stride ${LATENT_LAYER_STRIDE} \
    --latent_kl_weight ${LATENT_KL_WEIGHT} \
    --latent_sparsity_weight ${LATENT_SPARSITY_WEIGHT} \
    --latent_entropy_weight ${LATENT_ENTROPY_WEIGHT} \
    --latent_volume_weight ${LATENT_VOLUME_WEIGHT} \
    --latent_num_components ${LATENT_NUM_COMPONENTS} \
    --latent_min_sigma ${LATENT_MIN_SIGMA} \
    --latent_bias_alpha ${LATENT_BIAS_ALPHA}
