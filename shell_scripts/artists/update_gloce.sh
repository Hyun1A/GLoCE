#!/bin/bash

CONFIG="./configs/train_artist/config_train_gloce.yaml"

N_TOKENS="8"
GATE_RANK_LIST=( "1" )
UPDATE_RANK_LIST=( "1" )
DEGEN_RANK="1"
THRESH=1.5
ST_STACK=0 # (0-20, 0-50, 10-50)
END_STACK=50
N_GEN=3 # (1, 3, 8, 16, 32)

FIND_MODULE_LIST=( "unet_ca_out" )
DOMAIN="artist"
NUM_TARGETS=100

for FIND_MODULE in "${FIND_MODULE_LIST[@]}"; do
  for GATE_RANK in "${GATE_RANK_LIST[@]}"; do
    for UPDATE_RANK in "${UPDATE_RANK_LIST[@]}"; do
      for var in {0..99}; do
        echo "Running with UPDATE_RANK=${UPDATE_RANK}, DEGEN_RANK=${DEGEN_RANK}, GATE_RANK=${GATE_RANK}"
        CUDA_VISIBLE_DEVICES=0 python ./update/update_gloce.py \
          --config_file ${CONFIG} \
          --gate_rank ${GATE_RANK} \
          --update_rank ${UPDATE_RANK} \
          --degen_rank ${DEGEN_RANK} \
          --n_tokens ${N_TOKENS} \
          --use_emb_cache True \
          --find_module_name ${FIND_MODULE} \
          --n_target_concepts 1 \
          --n_anchor_concepts 3 \
          --tar_concept_idx ${var} \
          --thresh ${THRESH} \
          --st_timestep ${ST_STACK} \
          --end_timestep ${END_STACK} \
          --n_generation_per_concept ${N_GEN} \
          --param_cache_path "./importance_cache/org_comps/sd_v1.4/" \
          --emb_cache_path "./importance_cache/text_embs/sd_v1.4/${DOMAIN}_${NUM_TARGETS}" \
          --save_path "output/${DOMAIN}_${NUM_TARGETS}/${FIND_MODULE}/ur${UPDATE_RANK}_dr${DEGEN_RANK}_gr${GATE_RANK}_st${ST_STACK}_end${END_STACK}_n_gen${N_GEN}_th${THRESH}" \
          --buffer_path "./importance_cache/buffers/${DOMAIN}_${NUM_TARGETS}/sd_v1.4/${FIND_MODULE}/ur${UPDATE_RANK}_dr${DEGEN_RANK}_gr${GATE_RANK}_st${ST_STACK}_end${END_STACK}_n_gen${N_GEN}_th${THRESH}" \
          --emb_cache_fn "text_emb_cache.pt" \
          --last_layer "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0"
      done
    done
  done
done