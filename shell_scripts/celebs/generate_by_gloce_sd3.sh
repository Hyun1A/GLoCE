#!/bin/bash
# Generation with GLoCE-erased SD3 (omit --model_paths for the original model)
# usage: bash shell_scripts/celebs/generate_by_gloce_sd3.sh

GATE_RANK="4"
UPDATE_RANK="16"
DEGEN_RANK="2"
THRESH=2.5
ST_STACK=3
END_STACK=8
N_GEN=8

FIND_MODULE="sd3_block_context"
DOMAIN="celeb"

CKPT_ROOT="output/${DOMAIN}_sd3/${FIND_MODULE}/ur${UPDATE_RANK}_dr${DEGEN_RANK}_gr${GATE_RANK}_st${ST_STACK}_end${END_STACK}_gen${N_GEN}_th${THRESH}"

# erased model (barack obama removed, bob marley preserved)
CUDA_VISIBLE_DEVICES=0 python ./generate/generate_by_gloce_sd3.py \
  --model_paths "${CKPT_ROOT}/barack_obama" \
  --find_module_name ${FIND_MODULE} \
  --gate_rank ${GATE_RANK} \
  --update_rank ${UPDATE_RANK} \
  --degen_rank ${DEGEN_RANK} \
  --last_layer "transformer_blocks.22" \
  --st_step 3 --n_step 28 \
  --prompts "Barack Obama alongside Bob Marley in the shot" \
  --seeds 1,2,3,4,5,6,7,8 \
  --steps 28 --guidance 7.0 --size 512 \
  --out_dir "generated_images/${DOMAIN}_sd3/obama_erased"

# original SD3 for comparison
CUDA_VISIBLE_DEVICES=0 python ./generate/generate_by_gloce_sd3.py \
  --prompts "Barack Obama alongside Bob Marley in the shot" \
  --seeds 1,2,3,4,5,6,7,8 \
  --steps 28 --guidance 7.0 --size 512 \
  --out_dir "generated_images/${DOMAIN}_sd3/obama_base"
