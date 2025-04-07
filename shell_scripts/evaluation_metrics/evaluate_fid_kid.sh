################################################################
############## options for evaluating scores ###################

### dir1 / dir2: Directory for calculating KID/FID scores
### mode: Choose between KID (Kernel Inception Distance) and FID (Frechet Inception Distance).
###       KID is known to be more stable and reliable (unbiased) even with fewer samples.

############## options for evaluating scores ###################
################################################################

IMG_DIR1=image_save_folder
IMG_DIR2=image_save_folder
MODE=evaluation_mode ## kid / fid / KID / FID

python ./metrics/evaluate_fid_score.py \
    --dir1 ${IMG_DIR1} \
    --dir2 ${IMG_DIR2} \
    --mode ${PROMPT_CSV} 