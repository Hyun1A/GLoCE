################################################################
############## options for evaluating scores ###################

### image_dir: Directory for calculating CLIP scores
### prompts_path: Prompt file for generating images with a diffusion model.

############## options for evaluating scores ###################
################################################################

IMG_DIR=image_save_folder
PROMPT_CSV=prompt_save_path

python ./metrics/evaluate_clip_score.py \
    --image_dir ${IMG_DIR} \
    --prompts_path ${PROMPT_CSV}