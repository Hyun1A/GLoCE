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

def infer_with_gloce_org(
        config: GenerationConfig,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        v2: bool = False,
        precision: str = "fp32",
    ):

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
                print(os.path.isfile(config.save_path.format(prompt[:100].replace(" ", "_"), row['evaluation_seed'], "0")))
                if os.path.isfile(config.save_path.format(prompt[:100].replace(" ", "_"), row['evaluation_seed'], "0")):
                    print(row)
                    print('cont')                      
                    continue

                
            else:
                print(os.path.isfile(config.save_path.format(prompt.replace(" ", "_"), row['evaluation_seed'], "0")))
                if os.path.isfile(config.save_path.format(prompt.replace(" ", "_"), row['evaluation_seed'], "0")):
                    print(row)
                    print('cont\n')  
                    continue


            guidance_scale = row["evaluation_guidance"] if hasattr(row, "evaluation_guidance") else config.guidance_scale

            seed_everything(row['evaluation_seed'])
            
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
            
            folder = Path(config.save_path.format(prompt.replace(" ", "_"), "0", "0")).parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images):
                image.save(
                    config.save_path.format(
                        prompt.replace(" ", "_"), row['evaluation_seed'], i
                    )
                )
            






def main(args):
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

    infer_with_gloce_org(
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
        default="./",
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
