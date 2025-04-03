################################################################
############## options for evaluating scores ###################

### image_folder: Directory for GIPHY celebrity detector
### save_excel_path: Save path for GCD result

############## options for evaluating scores ###################
################################################################

IMG_DIR=image_save_folder
SAVE_EXCEL_PATH=result_save_path

python ./metrics/eevaluate_giphy_score.py \
    --image_folder ${IMG_DIR} \
    --save_excel_path ${SAVE_EXCEL_PATH}