#!/bin/bash

N_TOKENS="8"

GEN_CONFIG_LIST=("configs/gen_artist/config_artist_erased_fidel_camille_pissarro.yaml") 

# options for domains:
# configs/gen_artist/config_actor.yaml 
# configs/gen_artist/config_artist_erased_fidel_camille_pissarro.yaml 
# configs/gen_artist/config_artist_erased_fidel_canaletto.yaml 
# configs/gen_artist/config_artist_erased_fidel_claude_monet.yaml 
# configs/gen_artist/config_artist_erased.yaml 
# configs/gen_artist/config_artist_remained.yaml 
# configs/gen_artist/config_character.yaml 
# configs/gen_artist/config_coco_30k.yaml



GATE_RANK_LIST=( "1" )
UPDATE_RANK_LIST=( "1" )
DEGEN_RANK="1"
ETA_LIST=("1.0") # "3.0" "5.0" "10.0")
THRESH=2.5
ST_STACK=0  # (0-20, 0-50, 10-50)
END_STACK=50
N_GEN=3 # (1, 3, 8, 16, 32)

FIND_MODULE_LIST=("unet_ca_out")
DOMAIN="artist"
NUM_TARGETS=100
ST=0
END=5000

for FIND_MODULE in "${FIND_MODULE_LIST[@]}"; do
  for GATE_RANK in "${GATE_RANK_LIST[@]}"; do
    for UPDATE_RANK in "${UPDATE_RANK_LIST[@]}"; do
      for ETA in "${ETA_LIST[@]}"; do
        for GEN_CONFIG in "${GEN_CONFIG_LIST[@]}"; do
          echo "Generating with UPDATE_RANK=${UPDATE_RANK}, DEGEN_RANK=${DEGEN_RANK}, GATE_RANK=${GATE_RANK}, ETA=${ETA}"
          echo "generation config:" ${GEN_CONFIG}
          CUDA_VISIBLE_DEVICES=0 python ./generate/generate_by_gloce.py --config ${GEN_CONFIG} \
              --model_path "./output/${DOMAIN}_${NUM_TARGETS}/${FIND_MODULE}/ur${UPDATE_RANK}_dr${DEGEN_RANK}_gr${GATE_RANK}_st${ST_STACK}_end${END_STACK}_n_gen${N_GEN}_th${THRESH}" \
              --save_env "${DOMAIN}_${NUM_TARGETS}/${FIND_MODULE}/ur${UPDATE_RANK}_dr${DEGEN_RANK}_gr${GATE_RANK}_st${ST_STACK}_end${END_STACK}_n_gen${N_GEN}_th${THRESH}_eta${ETA}" \
              --gate_rank ${GATE_RANK} \
              --update_rank ${UPDATE_RANK} \
              --degen_rank ${DEGEN_RANK} \
              --eta ${ETA} \
              --st_timestep ${ST_STACK} \
              --find_module_name ${FIND_MODULE} \
              --gen_domain ${DOMAIN} \
              --st_prompt_idx ${ST} \
              --end_prompt_idx ${END} \
              --last_layer "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0"
        done
      done
    done
  done
done
