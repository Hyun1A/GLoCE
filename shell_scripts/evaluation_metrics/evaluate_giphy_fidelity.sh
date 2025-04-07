################################################################
########## options for evaluating GIPHY fidelity ###############

### image_folder: Directory for GIPHY celebrity detector
### save_excel_path: Save path for GCD result

########## options for evaluating GIPHY fidelity ###############
################################################################

IMG_DIR=/data/sjlim_diff/MACE/result_GLoCE/celeb_50/fidelity_celeb-celeb
SAVE_EXCEL_PATH=/data/sjlim_diff/MACE/result_GLoCE/celeb_50/fidelity_celeb-celeb

python ./metrics/evaluate_giphy_fidelity.py \
    --image_folder ${IMG_DIR} \
    --save_excel_path ${SAVE_EXCEL_PATH}