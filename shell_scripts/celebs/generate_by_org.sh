GEN_CONFIG_LIST=( "configs/gen_celeb/config_celeb_erased_fidel_elon_musk.yaml" )

for GEN_CONFIG in "${GEN_CONFIG_LIST[@]}"; do
    echo "Generating iamges by original model"
    echo "generation config:" ${GEN_CONFIG}
    CUDA_VISIBLE_DEVICES=0 python ./generate/generate_by_org.py --config ${GEN_CONFIG} \
        --save_env "sd_org" \
        --gen_domain "celeb"
done
