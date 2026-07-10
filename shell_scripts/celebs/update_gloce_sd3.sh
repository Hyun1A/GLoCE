#!/bin/bash
# GLoCE update (training-free parameter estimation) for Stable Diffusion 3
# usage: bash shell_scripts/celebs/update_gloce_sd3.sh <tar_concept_idx>
#   e.g. 1 = barack obama, 301 = queen elizabeth
#   (see configs/train_celeb/prompt_train_gloce_sd3_target.yaml)

CONFIG="./configs/train_celeb/config_train_gloce_sd3.yaml"

N_TOKENS="8"
GATE_RANK="4"
UPDATE_RANK="16"
DEGEN_RANK="2"
THRESH=2.5
ST_STACK=3   # transformer forwards 3-8 of the 28-step schedule
END_STACK=8
N_GEN=8

FIND_MODULE="sd3_block_context"
DOMAIN="celeb"

VAR=${1:-1}

echo "Running with UPDATE_RANK=${UPDATE_RANK}, DEGEN_RANK=${DEGEN_RANK}, GATE_RANK=${GATE_RANK}, tar_concept_idx=${VAR}"
CUDA_VISIBLE_DEVICES=0 python ./update/update_gloce_sd3.py \
  --config_file ${CONFIG} \
  --gate_rank ${GATE_RANK} \
  --update_rank ${UPDATE_RANK} \
  --degen_rank ${DEGEN_RANK} \
  --n_tokens ${N_TOKENS} \
  --use_emb_cache True \
  --find_module_name ${FIND_MODULE} \
  --n_target_concepts 1 \
  --n_anchor_concepts 3 \
  --tar_concept_idx ${VAR} \
  --thresh ${THRESH} \
  --st_timestep ${ST_STACK} \
  --end_timestep ${END_STACK} \
  --n_generation_per_concept ${N_GEN} \
  --param_cache_path "./importance_cache/org_comps/sd_v3" \
  --emb_cache_path "./importance_cache/text_embs/sd_v3" \
  --emb_cache_fn "text_emb_cache_anchor3.pt" \
  --save_path "output/${DOMAIN}_sd3/${FIND_MODULE}/ur${UPDATE_RANK}_dr${DEGEN_RANK}_gr${GATE_RANK}_st${ST_STACK}_end${END_STACK}_gen${N_GEN}_th${THRESH}" \
  --buffer_path "./importance_cache/buffers/${DOMAIN}_sd3/${FIND_MODULE}/ur${UPDATE_RANK}_dr${DEGEN_RANK}_gr${GATE_RANK}_st${ST_STACK}_end${END_STACK}_gen${N_GEN}_th${THRESH}" \
  --last_layer "transformer_blocks.22"
