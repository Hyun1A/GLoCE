import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from typing import Literal
import sys
import random, os
import numpy as np


sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

from src.configs.generation_config import load_config_from_yaml, GenerationConfig
from src.configs.config import parse_precision
from src.engine import train_util
from src.models import model_util
from src.models.gloce import(
    GLoCELayerOutProp,
    GLoCENetworkOutProp,
)
import src.engine.gloce_util as gloce_util


from src.models.merge_gloce import *
device = torch.device('cuda:0')
torch.cuda.set_device(device)

UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"



def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def flush():
    torch.cuda.empty_cache()
    gc.collect()

def infer_with_gloce(
        args,
        model_path: list[str],
        config: GenerationConfig,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        v2: bool = False,
        precision: str = "fp32",
    ):

    model_paths = model_path
    weight_dtype = parse_precision(precision)
        
    # load the pretrained SD
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()




    ####################################################################################
    ############################## register org modules ################################              
    module_types = []
    module_names = []
    org_modules_all = []
    module_name_list_all = []
    
    for find_module_name in args.find_module_name:
        module_name, module_type = gloce_util.get_module_name_type(find_module_name)
        param_cache_path = f"./importance_cache/org_comps/sd_v1.4" 
            
        org_modules, module_name_list = gloce_util.get_modules_list(unet, text_encoder, \
                                                    find_module_name, module_name, module_type)

        module_names.append(module_name)
        module_types.append(module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)
    ############################## register org modules ################################              
    ####################################################################################


    cpes, metadatas = zip(*[
        load_state_dict(model_path, weight_dtype) for model_path in model_paths
    ])
        
    # check if CPEs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    # get the erased concept
    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    # print(f"Erased prompts: {erased_prompts}")
    # print(metadatas[0])


    network = GLoCENetworkOutProp(
        unet,
        text_encoder,
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=GLoCELayerOutProp,
        degen_rank=args.degen_rank,
        gate_rank=args.gate_rank,
        update_rank=args.update_rank,
        n_concepts=len(model_paths),
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names = args.find_module_name,
        last_layer=args.last_layer,
        st_step=args.st_timestep,
    ).to(device, dtype=weight_dtype)  
    


    for n_concept in range(len(cpes)):
        print("loaded concepts:", n_concept+1)
        for idx, (k,m) in enumerate(network.named_modules()): #.items():
            if m.__class__.__name__ == "GLoCELayerOutProp":
                m.eta = args.eta
                        
                for k_child, m_child in m.named_children():
                    module_name = f"{k}.{k_child}"
                    if ("lora_update" in k_child) or ("lora_degen" in k_child):
                        m_child.weight.data[n_concept] = cpes[n_concept][module_name+'.weight']
                        print(f"{module_name+'.weight':100}", cpes[n_concept][module_name+'.weight'].shape)

                    elif "bias" in k_child:
                        m_child.weight.data[:,n_concept:n_concept+1,:] = cpes[n_concept][module_name+'.weight']
                        print(f"{module_name+'.weight':100}", cpes[n_concept][module_name+'.weight'].shape)

                    elif "selector" in k_child:
                        m_child.select_weight.weight.data[n_concept] = cpes[n_concept][module_name+'.select_weight.weight'].squeeze(0)
                        m_child.select_mean_diff.weight.data[n_concept] = cpes[n_concept][module_name+'.select_mean_diff.weight'].squeeze(0)
                        m_child.imp_center[n_concept] = cpes[n_concept][module_name+'.imp_center']
                        m_child.imp_slope[n_concept] = cpes[n_concept][module_name+'.imp_slope']

                        # print(f"{module_name+'.select_weight.weight':100}", cpes[n_concept][module_name+'.select_weight.weight'].shape)


    network.to(device, dtype=weight_dtype)  
    
    network.eval()
    
    print(config.save_path)
    
    promptDf = pd.read_csv(config.prompt_path)

    with torch.no_grad():
        for p_idx, (k, row) in enumerate(promptDf.iterrows()):
            
            if (p_idx < config.st_prompt_idx) or (p_idx > config.end_prompt_idx):
                continue
            
            
            prompt = str(row['prompt'])
            prompt += config.unconditional_prompt
            print(f"{p_idx}, Generating for prompt: {prompt}")
            prompt_embeds, prompt_tokens = train_util.encode_prompts(
                tokenizer, text_encoder, [prompt], return_tokens=True
                )
            

            if args.gen_domain == "explicit":
                print(os.path.isfile(config.save_path.format(prompt[:100].replace(" ", "-"), row['evaluation_seed'], "0")))
                if os.path.isfile(config.save_path.format(prompt[:100].replace(" ", "-"), row['evaluation_seed'], "0")):
                    print(row)
                    print('cont')                      
                    continue

            else:
                print(os.path.isfile(config.save_path.format(prompt.replace(" ", "-"), row['evaluation_seed'], "0")))
                if os.path.isfile(config.save_path.format(prompt.replace(" ", "-"), row['evaluation_seed'], "0")):
                    print(row)
                    print('cont\n')  
                    continue


            guidance_scale = row["evaluation_guidance"] if hasattr(row, "evaluation_guidance") else config.guidance_scale

            seed_everything(row['evaluation_seed'])
            
            with network:
                images = pipe(
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.cuda.manual_seed(row['evaluation_seed']),
                    num_images_per_prompt=config.generate_num,
                    prompt_embeds=prompt_embeds,
                ).images

            if len(prompt) > 100:
                prompt = prompt[:100]
            
            folder = Path(config.save_path.format(prompt.replace(" ", "-"), "0", "0")).parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images):
                image.save(
                    config.save_path.format(
                        prompt.replace(" ", "-"), row['evaluation_seed'], i
                    )
                )
            






def main(args):
    concepts_ckpt = []
    
    # ckpt_path = f"gate_rank{args.gate_rank}_update_rank{args.update_rank}_lamb{args.lamb}.safetensors"
    ckpt_path = "ckpt.safetensors"

    ckpts = os.listdir(args.model_path[0])
    for ckpt in ckpts:
        if os.path.isfile(os.path.join(args.model_path[0],ckpt, ckpt_path)):
            concepts_ckpt.append(os.path.join(args.model_path[0],ckpt, ckpt_path))


    model_path = [Path(lp) for lp in concepts_ckpt]
    
    generation_config = load_config_from_yaml(args.config)

    if args.st_prompt_idx != -1:
        generation_config.st_prompt_idx = args.st_prompt_idx
    if args.end_prompt_idx != -1:
        generation_config.end_prompt_idx = args.end_prompt_idx
    if args.gate_rank != -1:
        generation_config.gate_rank = args.gate_rank
    
    
    generation_config.save_path = os.path.join("/".join(generation_config.save_path.split("/")[:-3]), args.save_env, "/".join(generation_config.save_path.split("/")[-2:]))

    args.find_module_name = args.find_module_name.split(",")
    if args.find_module_name.__class__ == str:
        args.find_module_name = [args.find_module_name]

    infer_with_gloce(
        args,
        model_path,
        generation_config,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generation.yaml",
        help="Base configs for image generation.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        nargs="*",
        help="CPE model to use.",
    )
    # model configs
    parser.add_argument(
        "--base_model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for the base model.",
    )
    parser.add_argument(
        "--save_env",
        type=str,
        default="",
        help="Precision for the base model.",
    )    
    
    parser.add_argument(
        "--st_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--end_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--degen_rank",
        type=int,
        default=-1,
    )
    
    
    parser.add_argument(
        "--gate_rank",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--update_rank",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--st_timestep",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--end_timestep",
        type=int,
        default=-1,
    )


    parser.add_argument(
        "--lamb",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--find_module_name",
        type=str,
        default="unet_ca",
    )

    parser.add_argument(
        "--gen_domain",
        type=str,
        default="celeb",
    )


    parser.add_argument(
        "--last_layer",
        type=str,
        default="",
    )



    args = parser.parse_args()

    main(args)
