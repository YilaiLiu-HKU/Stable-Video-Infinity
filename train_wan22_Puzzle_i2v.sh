#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,4}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29618}"

cd /data/yilai/svi_wan22

CKPT_DIR="/data/yilai/video_model/models/Wan2.2-I2V-A14B"
DATA_ROOT="/data/yilai/sample_dataset"
CANDIDATE_CSV="${CANDIDATE_CSV:-${DATA_ROOT}/candidate_groups.csv}"
CHARACTER_LISTS_DIR="${CHARACTER_LISTS_DIR:-${DATA_ROOT}/character_lists}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/video}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/yilai/svi_wan22/experiments/wan22_storymem_i2v}"
EXP_PREFIX_BASE="${EXP_PREFIX_BASE:-wan22_i2v_storymem_sparse_ctx_single_noise}"
ROLE_TOKEN_SELECTION_MODE="${ROLE_TOKEN_SELECTION_MODE:-two_role_diff}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-$(python3 - <<'PY'
import os
value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
parts = [x for x in value.split(",") if x.strip()]
print(len(parts) if parts else 1)
PY
)}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
LOW_LORA_INIT_PATH="${LOW_LORA_INIT_PATH:-}"
HIGH_LORA_INIT_PATH="${HIGH_LORA_INIT_PATH:-}"

mkdir -p "${OUTPUT_ROOT}"

common_args=(
  --ckpt_dir "${CKPT_DIR}"
  --story_root ""
  --candidate_groups_csv "${CANDIDATE_CSV}"
  --character_lists_dir "${CHARACTER_LISTS_DIR}"
  --video_root "${VIDEO_ROOT}"
  --max_epochs "${MAX_EPOCHS}"
  --num_train_gpus "${NUM_TRAIN_GPUS}"
  --num_nodes 1
  --train_architecture lora
  --lora_rank 64
  --lora_alpha 64
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
  --noise_domain_boundary_ratio 0.9
  --learning_rate 1e-4
  --latent_dim 16
  --patch_dim 5120
  --projector_bottleneck 256
  --num_frames 81
  --num_motion_frames 5
  --ref_pad_num 1
  --use_gradient_checkpointing
  --use_gradient_checkpointing_offload
  --aggressive_vram_optimization
  --training_strategy "ddp_find_unused_parameters_true"
  --model_slice_mode "none"
  --tp_size 1
  --extract_single_timestep_align_train
  --use_train_weights_for_extract_and_probe
  --precompute_image_emb
  --precompute_image_emb_strict
  --offload_image_encoder_after_extraction
  --no-keep_image_encoder_on_gpu
  --memory_injection_mode "context_only"
  --sparse_role_memory_layer_idx 7
  --sparse_role_memory_num_heads 8
  --sparse_role_memory_head_dim 128
  --train_stage stage2
  --sparse_role_memory_rope_dim 256
  --sparse_role_memory_use_half_role_heads
  --sparse_role_memory_feature_source "attn_out"
  --sparse_role_memory_init_scale 0.1
  --sparse_role_memory_time_gate
  --role_token_selection_mode "${ROLE_TOKEN_SELECTION_MODE}"
  --extract_layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  --use_attn_score_selection
  --top_visual_tokens 0.1
  --token_weight 1
  --cfg_scale_extraction 5.0
  --max_memory_characters 2
  --neighbor_filter_kernel 5
  --max_memory_tokens_per_character 64
  --memory_bank_percents "0.85,0.60,0.35,0.12"
  --prompt_drop_prob 0.0
  --memory_drop_prob 0.0
  --image_condition_people_pixelate_prob 0.2
  --image_condition_people_pixelate_conf 0.25
  --image_condition_people_pixelate_block_size 12
  --image_condition_people_pixelate_mask_dilate_kernel 9
  --checkpoint_save_every_n_epochs 0
  --checkpoint_save_every_n_steps 500
  --seed -1
  --perf_log_interval 0
)

run_phase() {
  local phase_name="$1"
  local train_noise_domain="$2"
  local char_attn_scope="$3"
  local enable_sparse_role_memory="$4"
  local max_steps="$5"
  local output_dir="$6"
  local exp_prefix="$7"
  local pretrained_lora_path="${8:-}"

  mkdir -p "${output_dir}"

  echo "=========================================="
  echo "Wan2.2 I2V StoryMem Training"
  echo "=========================================="
  echo "PHASE=${phase_name}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "CANDIDATE_CSV=${CANDIDATE_CSV}"
  echo "CHARACTER_LISTS_DIR=${CHARACTER_LISTS_DIR}"
  echo "VIDEO_ROOT=${VIDEO_ROOT}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "EXP_PREFIX=${exp_prefix}"
  echo "TRAIN_NOISE_DOMAIN=${train_noise_domain}"
  echo "CHAR_ATTN_NOISE_SCOPE=${char_attn_scope}"
  echo "ENABLE_SPARSE_ROLE_MEMORY=${enable_sparse_role_memory}"
  echo "PRETRAINED_LORA_PATH=${pretrained_lora_path:-<none>}"
  echo "=========================================="

  phase_args=(
    --output_path "${output_dir}"
    --exp_prefix "${exp_prefix}"
    --max_steps "${max_steps}"
    --train_noise_domain "${train_noise_domain}"
    --char_attn_noise_scope "${char_attn_scope}"
  )

  if [[ "${enable_sparse_role_memory}" == "1" ]]; then
    phase_args+=(--enable_sparse_role_memory_attn)
  else
    phase_args+=(--no-enable_sparse_role_memory_attn)
  fi

  if [[ -n "${pretrained_lora_path}" ]]; then
    phase_args+=(--pretrained_lora_path "${pretrained_lora_path}")
  fi

  torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" --master-port="${MASTER_PORT}" train_Puzzle.py \
    "${common_args[@]}" \
    "${phase_args[@]}"
}

run_phase \
  "low_noise" \
  "low_noise" \
  "low_noise" \
  "1" \
  "1600" \
  "${OUTPUT_ROOT}/low_noise" \
  "${EXP_PREFIX_BASE}_low_noise" \
  "${LOW_LORA_INIT_PATH}"

run_phase \
  "high_noise" \
  "high_noise" \
  "high_noise" \
  "0" \
  "500" \
  "${OUTPUT_ROOT}/high_noise" \
  "${EXP_PREFIX_BASE}_high_noise" \
  "${HIGH_LORA_INIT_PATH}"

run_phase \
  "high_noise" \
  "high_noise" \
  "high_noise" \
  "1" \
  "500" \
  "${OUTPUT_ROOT}/high_noise_with_char_attn_ablation" \
  "${EXP_PREFIX_BASE}_high_noise_with_char_attn_ablation" \
  "${HIGH_LORA_INIT_PATH}"

run_phase \
  "low_noise" \
  "low_noise" \
  "low_noise" \
  "0" \
  "1600" \
  "${OUTPUT_ROOT}/low_noise_without_char_attn_ablation" \
  "${EXP_PREFIX_BASE}_low_noise_without_char_attn_ablation" \
  "${LOW_LORA_INIT_PATH}"
