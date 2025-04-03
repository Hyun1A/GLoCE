################################################################
############## options for evaluating scores ###################

### image_dir: Directory for calculating I2P (nudity) test
### prompts_path: Prompt file for generating images with a diffusion model.

############## options for evaluating scores ###################
################################################################

IMG_DIR=image_save_folder
SAVE_EXCEL_PATH=prompt_save_path

python ./metrics/evaluate_I2P.py \
    --image_folder ${IMG_DIR} \
    --save_excel_path ${SAVE_EXCEL_PATH}