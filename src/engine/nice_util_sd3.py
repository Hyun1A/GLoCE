# ref: 
# - https://github.com/p1atdev/LECO/blob/main/train_util.py

from typing import Optional, Union

import os, sys
import ast
import importlib
import math

import pandas as pd
import random
import numpy as np

from tqdm import tqdm
from PIL import Image

import torch
from torch.optim import Optimizer
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin, DiffusionPipeline
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from src.models.model_util import SDXL_TEXT_ENCODER_TYPE
import src.engine.train_util_sd3 as train_util_SD3

from .gloce_register_buffer_sd3 import *


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def get_module_name_type(find_module_name):
    if find_module_name == "unet_ca":
        module_type = "Linear"
        module_name = "attn2"

    elif find_module_name == "unet_ca_kv":
        module_type = "Linear"
        module_name = "attn2"

    elif find_module_name == "unet_ca_out":
        module_type = "Linear"
        module_name = "attn2"
        
    elif find_module_name == "unet_sa_out":
        module_type = "Linear"
        module_name = "attn1"

    elif find_module_name == "unet_sa":
        module_type = "Linear"
        module_name = "attn1"

    elif find_module_name == "unet_conv2d":
        module_type = "Conv2d"
        module_name = "conv2d"           

    elif find_module_name == "unet_misc":
        module_type = "Linear"
        module_name = "misc"

    elif find_module_name == "te_attn":        
        module_type = "Linear"
        module_name = "self_attn"
        
    elif find_module_name == "sd3_self_out":
        module_type = "Linear"
        module_name = "attn"

    elif find_module_name == "sd3_add_out":
        module_type = "Linear"
        module_name = "attn"

    elif find_module_name == "sd3_block_context":
        module_type = "JointTransformerBlock"
        module_name = "transformer_blocks"
        
    elif find_module_name == "sd3_transformer_block":
        module_type = "JointTransformerBlock"
        module_name = "transformer_blocks"

    else:
        module_type = "Linear"
        module_name = "mlp.fc"

    return module_name, module_type

def get_modules_list(unet, text_encoder, find_module_name, module_name, module_type):
    org_modules = dict()
    module_name_list = []

    if find_module_name == "unet_ca_out":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_out" in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    elif find_module_name == "unet_ca_kv":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_k" in n) or (module_name+".to_v" in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    elif find_module_name == "unet_sa_out":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_out" in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    elif "unet" in find_module_name:
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if module_name == "misc":
                    if ("attn1" not in n) and ("attn2" not in n):
                        module_name_list.append(n)
                        org_modules[n] = m

                elif (module_name == "attn1") or (module_name == "attn2"): 
                    if module_name in n:
                        module_name_list.append(n)
                        org_modules[n] = m

                else:
                    module_name_list.append(n)
                    org_modules[n] = m
                    
    if find_module_name == "sd3_self_out":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_out" in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    elif find_module_name == "sd3_add_out":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_add_out" in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    elif find_module_name == "sd3_block_context":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name in n) and (not m.context_pre_only):
                    module_name_list.append(n)
                    org_modules[n] = m
                    
    elif find_module_name == "sd3_transformer_block":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    else:
        for n,m in text_encoder.named_modules():
            if m.__class__.__name__ == module_type:       
                if module_name in n:
                    module_name_list.append(n)
                    org_modules[n] = m

    return org_modules, module_name_list

def load_model_sv_cache(find_module_name, param_cache_path, device, org_modules):
    
    if os.path.isfile(f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt"):
        print("load precomputed svd for original models ....")

        param_vh_cache_dict = torch.load(f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt", map_location=torch.device(device)) 
        param_s_cache_dict = torch.load(f"{param_cache_path}/s_cache_dict_{find_module_name}.pt", map_location=torch.device(device))

    else:
        print("compute svd for original models ....")

        param_vh_cache_dict = dict()
        param_s_cache_dict = dict()

        for idx_mod, (k,m) in enumerate(org_modules.items()):
            print(idx_mod, k)
            if m.__class__.__name__ == "Linear":
                U,S,Vh = torch.linalg.svd(m.weight, full_matrices=False) 
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()        

            elif m.__class__.__name__ == "Conv2d":
                module_weight_flatten = m.weight.view(m.weight.size(0), -1)

                U,S,Vh = torch.linalg.svd(module_weight_flatten, full_matrices=False) 
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()                

        os.makedirs(param_cache_path, exist_ok=True)
        torch.save(param_vh_cache_dict, f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt")
        torch.save(param_s_cache_dict, f"{param_cache_path}/s_cache_dict_{find_module_name}.pt")

    return param_vh_cache_dict, param_s_cache_dict


@torch.no_grad()
def prepare_text_embedding_token(args, config, prompts_target, prompts_anchor, prompts_update, pipe, train_util, DEVICE_CUDA,\
                                emb_cache_path, emb_cache_fn,
                                n_avail_tokens=8, n_anchor_concepts=5):
    # Prepare for text embedding token
    prompt_scripts_path = config.scripts_file


    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list = prompt_scripts_df['prompt'].to_list()
    replace_word = config.replace_word

    if replace_word == "artist":
        prmpt_temp_sel_base = f"An image in the style of {replace_word}" 
        # prmpt_temp_sel_base = replace_word
    elif replace_word == "celeb":
        # prmpt_temp_sel_base = f"A face of {replace_word}"
        prmpt_temp_sel_base = replace_word
    elif replace_word == "explicit":
        prmpt_temp_sel_base = replace_word

    prompt_scripts_list.append( prmpt_temp_sel_base )



    if args.use_emb_cache and os.path.isfile(f"{emb_cache_path}/{emb_cache_fn}"):
        print("load pre-computed text emb cache...")
        emb_cache = torch.load(f"{emb_cache_path}/{emb_cache_fn}", map_location=torch.device(DEVICE_CUDA))
        
    else:
        # Prepare for sel basis simWords
        print("compute text emb cache...")

        simWords_target = [prompt.target for prompt in prompts_target]
        simWords_anchor = [prompt.target for prompt in prompts_anchor]
        simWords_update = [prompt.target for prompt in prompts_update]
        simWords_surrogate = [prompts_target[0].neutral]

        for simW_erase in simWords_target:
            simWords_anchor = [item.lower() for item in simWords_anchor if simW_erase not in item.lower()]



        prmpt_sel_base_surrogate = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_surrogate] 
        embeddings_surrogate_sel_base, embeddings_surrogate_sel_base_pooled = train_util.encode_prompts(pipe, prmpt_sel_base_surrogate)

        prmpt_sel_base_target = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_target] 
        embeddings_target_sel_base, embeddings_target_sel_base_pooled = train_util.encode_prompts(pipe, prmpt_sel_base_target)

        prmpt_sel_base_anchor = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_anchor] 
        len_anchor = len(prmpt_sel_base_anchor)
        text_encode_batch = 100
        simWords_anc_batch = [
            prmpt_sel_base_anchor[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_anchor) / text_encode_batch)))
        ]

        embeddings_anchor_sel_base = []
        embeddings_anchor_sel_base_pooled = []
        for simW_batch in simWords_anc_batch:
            emb_anc, emb_anc_pooled = train_util.encode_prompts(pipe, simW_batch)
            embeddings_anchor_sel_base.append(emb_anc)
            embeddings_anchor_sel_base_pooled.append(emb_anc_pooled)
        embeddings_anchor_sel_base = torch.cat(embeddings_anchor_sel_base, dim=0)
        embeddings_anchor_sel_base_pooled = torch.cat(embeddings_anchor_sel_base_pooled, dim=0)

        prmpt_sel_base_update = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_update]
        embeddings_update_sel_base, embeddings_update_sel_base_pooled = train_util.encode_prompts(pipe, prmpt_sel_base_update)  

        # Compute similarity
        embeddings_target_norm = embeddings_target_sel_base / embeddings_target_sel_base.norm(2, dim=-1, keepdim=True)
        embeddings_anchor_norm = embeddings_anchor_sel_base / embeddings_anchor_sel_base.norm(2, dim=-1, keepdim=True)

        similarity = torch.einsum("ijk,njk->inj", embeddings_target_norm[:, 1:(1 + n_avail_tokens), :],
                                    embeddings_anchor_norm[:, 1:(1 + n_avail_tokens), :]).mean(dim=2)

        # Select anchor celebs
        similarity = similarity.mean(dim=0)
        val_sorted, ind_sorted = similarity.sort()
        ind_sorted_list = ind_sorted.cpu().numpy().tolist()

        simWords_anchor = [simWords_anchor[sim_idx] for sim_idx in ind_sorted_list[-n_anchor_concepts:]]
        embeddings_anchor_sel_base = embeddings_anchor_sel_base[ind_sorted_list[-n_anchor_concepts:]]


        # simWords_anchor = [simWords_anchor[sim_idx] for sim_idx in ind_sorted_list[:n_anchor_concepts]]
        # embeddings_anchor_sel_base = embeddings_anchor_sel_base[ind_sorted_list[:n_anchor_concepts]]
        # embeddings_anchor_sel_base_pooled = embeddings_anchor_sel_base_pooled[ind_sorted_list[:n_anchor_concepts]]
        # embeddings_anchor_norm = embeddings_anchor_norm[ind_sorted_list[:n_anchor_concepts]]





        # Prepare for erasing token cache
        print("compute emb cache...")
        prompt_in_scripts_neutral = []
        embeddings_surrogate_cache = []
        embeddings_surrogate_cache_pooled = []
        for prompt_script in prompt_scripts_list:
            pr_in_script_ntl = prompt_script.replace(replace_word, prompts_target[0].neutral)
            pr_in_script_ntl = pr_in_script_ntl.replace(replace_word.lower(), prompts_target[0].neutral)
            prompt_in_scripts_neutral.append(pr_in_script_ntl)

        embeddings_ntl, embeddings_ntl_pooled = train_util.encode_prompts(pipe, prompt_in_scripts_neutral)
        embeddings_surrogate_cache.append(embeddings_ntl)
        embeddings_surrogate_cache_pooled.append(embeddings_ntl_pooled)

        embeddings_surrogate_cache = torch.cat(embeddings_surrogate_cache, dim=0)
        embeddings_surrogate_cache_pooled = torch.cat(embeddings_surrogate_cache_pooled, dim=0)
        

        embeddings_target_cache = []
        embeddings_target_cache_pooled = []
        prmpt_scripts_tar = []

        for simWord in simWords_target:
            for prompt_script in prompt_scripts_list:
                pr_in_script_tar = prompt_script.replace(replace_word, simWord)
                pr_in_script_tar = pr_in_script_tar.replace(replace_word.lower(), simWord)
                prmpt_scripts_tar.append(pr_in_script_tar)

        len_target = len(prmpt_scripts_tar)
        text_encode_batch = 100
        prmpt_scripts_tar_batch = [
            prmpt_scripts_tar[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_target) / text_encode_batch)))
        ]

        for prmpt_batch in prmpt_scripts_tar_batch:
            embeddings_tar, embeddings_tar_pooled = train_util.encode_prompts(pipe, prmpt_batch)
            embeddings_target_cache.append(embeddings_tar)
            embeddings_target_cache_pooled.append(embeddings_tar_pooled)

        embeddings_target_cache = torch.cat(embeddings_target_cache, dim=0)
        embeddings_target_cache_pooled = torch.cat(embeddings_target_cache_pooled, dim=0)

        # Prepare for anchoring token cache
        embeddings_anchor_cache = []
        embeddings_anchor_cache_pooled = []
        prmpt_scripts_anc = []

        for simWord in simWords_anchor:
            for prompt_script in prompt_scripts_list:
                pr_in_script_anc = prompt_script.replace(replace_word, simWord)
                pr_in_script_anc = pr_in_script_anc.replace(replace_word.lower(), simWord)
                prmpt_scripts_anc.append(pr_in_script_anc)

        len_anchor = len(prmpt_scripts_anc)
        text_encode_batch = 100
        prmpt_scripts_anc_batch = [
            prmpt_scripts_anc[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_anchor) / text_encode_batch)))
        ]
        for prmpt_batch in prmpt_scripts_anc_batch:
            embeddings_anc, embeddings_anc_pooled = train_util.encode_prompts(pipe, prmpt_batch)
            embeddings_anchor_cache.append(embeddings_anc)
            embeddings_anchor_cache_pooled.append(embeddings_anc_pooled)

        embeddings_anchor_cache = torch.cat(embeddings_anchor_cache, dim=0)
        embeddings_anchor_cache_pooled = torch.cat(embeddings_anchor_cache_pooled, dim=0)



        # Prepare for update token cache
        embeddings_update_cache = []
        embeddings_update_cache_pooled = []
        prmpt_scripts_upd = []

        for simWord in simWords_update:
            for prompt_script in prompt_scripts_list:
                pr_in_script_upd = prompt_script.replace(replace_word, simWord)
                pr_in_script_upd = pr_in_script_upd.replace(replace_word.lower(), simWord)
                prmpt_scripts_upd.append(pr_in_script_upd)

        len_update = len(prmpt_scripts_upd)
        text_encode_batch = 100
        prmpt_scripts_upd_batch = [
            prmpt_scripts_upd[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_update) / text_encode_batch)))
        ]

        for prmpt_batch in prmpt_scripts_upd_batch:
            embeddings_upd, embeddings_upd_pooled = train_util.encode_prompts(pipe, prmpt_batch)
            embeddings_update_cache.append(embeddings_upd)
            embeddings_update_cache_pooled.append(embeddings_upd_pooled)

        embeddings_update_cache = torch.cat(embeddings_update_cache, dim=0)
        embeddings_update_cache_pooled = torch.cat(embeddings_update_cache_pooled, dim=0)

        # Save emb cache
        emb_cache = {
            "embeddings_surrogate_cache": embeddings_surrogate_cache,
            "embeddings_target_cache": embeddings_target_cache,
            "embeddings_anchor_cache": embeddings_anchor_cache,
            "embeddings_update_cache": embeddings_update_cache,
            "embeddings_surrogate_sel_base": embeddings_surrogate_sel_base,
            "embeddings_target_sel_base": embeddings_target_sel_base,
            "embeddings_anchor_sel_base": embeddings_anchor_sel_base,
            "embeddings_update_sel_base": embeddings_update_sel_base,
            "prmpt_scripts_sur": prompt_in_scripts_neutral,
            "prmpt_scripts_tar": prmpt_scripts_tar,
            "prmpt_scripts_anc": prmpt_scripts_anc,
            "prmpt_scripts_upd": prmpt_scripts_upd,
            
            "embeddings_surrogate_cache_pooled": embeddings_surrogate_cache_pooled,
            "embeddings_target_cache_pooled": embeddings_target_cache_pooled,
            "embeddings_anchor_cache_pooled": embeddings_anchor_cache_pooled,
            "embeddings_update_cache_pooled": embeddings_update_cache_pooled,
            "embeddings_surrogate_sel_base_pooled": embeddings_surrogate_sel_base_pooled,
            "embeddings_target_sel_base_pooled": embeddings_target_sel_base_pooled,
            "embeddings_anchor_sel_base_pooled": embeddings_anchor_sel_base_pooled,
            "embeddings_update_sel_base_pooled": embeddings_update_sel_base_pooled,
        }


        os.makedirs(emb_cache_path, exist_ok=True)
        torch.save(emb_cache, f"{emb_cache_path}/{emb_cache_fn}")

    return emb_cache



@torch.no_grad()
def prepare_text_embedding_token_past(args, config, prompts_target, prompts_anchor, pipe, train_util, DEVICE_CUDA,\
                                emb_cache_path, emb_cache_fn,
                                n_avail_tokens=8, n_anchor_concepts=5):
    # Prepare for text embedding token
    prompt_scripts_path = config.scripts_file
    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list = prompt_scripts_df['prompt'].to_list()
    replace_word = config.replace_word
    prompt_scripts_list += [replace_word] * 1

    if args.use_emb_cache and os.path.isfile(f"{emb_cache_path}/{emb_cache_fn}"):
        print("load pre-computed text emb cache...")
        emb_cache = torch.load(f"{emb_cache_path}/{emb_cache_fn}", map_location=torch.device(DEVICE_CUDA))
        
    else:
        # Prepare for sel basis simWords
        print("compute text emb cache...")

        if replace_word == "explicit":
            simWords_target = [prompt.target for prompt in prompts_target]
            simWords_surrogate = [prompts_target[0].neutral]

            simWords_csv = pd.read_csv("./anchors/mass_surr_prompts_explicit.csv")
            simWords_anchor = list(simWords_csv.itertuples(index=False, name=None))
            simWords_anchor = [(a, b) for _, a, b in simWords_anchor]

            embeddings_surrogate_sel_base, embeddings_surrogate_sel_base_pooled = train_util.encode_prompts(pipe, simWords_surrogate)
            embeddings_target_sel_base, embeddings_target_sel_base_pooled = train_util.encode_prompts(pipe, simWords_target)
            embeddings_anchor_sel_base, embeddings_anchor_sel_base_pooled = train_util.encode_prompts(pipe, simWords_anchor)
            embeddings_target_norm = embeddings_target_sel_base / embeddings_target_sel_base.norm(2, dim=-1, keepdim=True)
            embeddings_anchor_norm = embeddings_anchor_sel_base / embeddings_anchor_sel_base.norm(2, dim=-1, keepdim=True)

        else:
            # Prepare text embeddings
            simWords_target = [prompt.target for prompt in prompts_target]
            simWords_anchor = [prompt.target for prompt in prompts_anchor]
            simWords_surrogate = [prompts_target[0].neutral]

            for simW_erase in simWords_target:
                simWords_anchor = [item for item in simWords_anchor if simW_erase not in item.lower()]

            embeddings_surrogate_sel_base, embeddings_surrogate_sel_base_pooled = train_util.encode_prompts(pipe, simWords_surrogate)
            embeddings_target_sel_base, embeddings_target_sel_base_pooled = train_util.encode_prompts(pipe, simWords_target)

            len_anchor = len(simWords_anchor)
            text_encode_batch = 100
            simWords_anc_batch = [
                simWords_anchor[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
                for batch_idx in range(int(math.ceil(float(len_anchor) / text_encode_batch)))
            ]

            embeddings_anchor_sel_base = []
            embeddings_anchor_sel_base_pooled = []
            for simW_batch in simWords_anc_batch:
                emb_anc, emb_anc_pooled = train_util.encode_prompts(pipe, simW_batch)
                embeddings_anchor_sel_base.append(emb_anc)
                embeddings_anchor_sel_base_pooled.append(emb_anc_pooled)
            embeddings_anchor_sel_base = torch.cat(embeddings_anchor_sel_base, dim=0)
            embeddings_anchor_sel_base_pooled = torch.cat(embeddings_anchor_sel_base_pooled, dim=0)

            # Compute similarity
            embeddings_target_norm = embeddings_target_sel_base / embeddings_target_sel_base.norm(2, dim=-1, keepdim=True)
            embeddings_anchor_norm = embeddings_anchor_sel_base / embeddings_anchor_sel_base.norm(2, dim=-1, keepdim=True)

            similarity = torch.einsum("ijk,njk->inj", embeddings_target_norm[:, 1:(1 + n_avail_tokens), :],
                                      embeddings_anchor_norm[:, 1:(1 + n_avail_tokens), :]).mean(dim=2)

            # Select anchor celebs
            similarity = similarity.mean(dim=0)
            val_sorted, ind_sorted = similarity.sort()
            ind_sorted_list = ind_sorted.cpu().numpy().tolist()

            simWords_anchor = [simWords_anchor[sim_idx] for sim_idx in ind_sorted_list[-n_anchor_concepts:]]
            embeddings_anchor_sel_base = embeddings_anchor_sel_base[ind_sorted_list[-n_anchor_concepts:]]
            embeddings_anchor_sel_base_pooled = embeddings_anchor_sel_base_pooled[ind_sorted_list[-n_anchor_concepts:]]
            embeddings_anchor_norm = embeddings_anchor_norm[ind_sorted_list[-n_anchor_concepts:]]



        # Prepare for erasing token cache
        print("compute emb cache...")
        prompt_in_scripts_neutral = []
        embeddings_surrogate_cache = []
        embeddings_surrogate_cache_pooled = []
        for prompt_script in prompt_scripts_list:
            pr_in_script_ntl = prompt_script.replace(replace_word, prompts_target[0].neutral)
            pr_in_script_ntl = pr_in_script_ntl.replace(replace_word.lower(), prompts_target[0].neutral)
            prompt_in_scripts_neutral.append(pr_in_script_ntl)

        embeddings_ntl, embeddings_ntl_pooled = train_util.encode_prompts(pipe, prompt_in_scripts_neutral)
        embeddings_surrogate_cache.append(embeddings_ntl)
        embeddings_surrogate_cache_pooled.append(embeddings_ntl_pooled)

        embeddings_surrogate_cache = torch.cat(embeddings_surrogate_cache, dim=0)
        embeddings_surrogate_cache_pooled = torch.cat(embeddings_surrogate_cache_pooled, dim=0)

        embeddings_target_cache = []
        embeddings_target_cache_pooled = []
        prmpt_scripts_tar = []

        if replace_word == "explicit":
            prmpt_scripts_tar = [simWord for simWord in simWords_target]
        else:
            for simWord in simWords_target:
                for prompt_script in prompt_scripts_list:
                    pr_in_script_tar = prompt_script.replace(replace_word, simWord)
                    pr_in_script_tar = pr_in_script_tar.replace(replace_word.lower(), simWord)
                    prmpt_scripts_tar.append(pr_in_script_tar)

        len_target = len(prmpt_scripts_tar)
        text_encode_batch = 100
        prmpt_scripts_tar_batch = [
            prmpt_scripts_tar[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_target) / text_encode_batch)))
        ]

        for prmpt_batch in prmpt_scripts_tar_batch:
            embeddings_tar, embeddings_tar_pooled = train_util.encode_prompts(pipe, prmpt_batch)
            embeddings_target_cache.append(embeddings_tar)
            embeddings_target_cache_pooled.append(embeddings_tar_pooled)

        embeddings_target_cache = torch.cat(embeddings_target_cache, dim=0)
        embeddings_target_cache_pooled = torch.cat(embeddings_target_cache_pooled, dim=0)

        # Prepare for anchoring token cache
        embeddings_anchor_cache = []
        embeddings_anchor_cache_pooled = []
        prmpt_scripts_anc = []

        if replace_word == "explicit":
            prmpt_scripts_anc = [simWord for simWord in simWords_anchor]
        else:
            for simWord in simWords_anchor:
                for prompt_script in prompt_scripts_list:
                    pr_in_script_anc = prompt_script.replace(replace_word, simWord)
                    pr_in_script_anc = pr_in_script_anc.replace(replace_word.lower(), simWord)
                    prmpt_scripts_anc.append(pr_in_script_anc)

        len_anchor = len(prmpt_scripts_anc)
        text_encode_batch = 100
        prmpt_scripts_anc_batch = [
            prmpt_scripts_anc[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_anchor) / text_encode_batch)))
        ]
        for prmpt_batch in prmpt_scripts_anc_batch:
            embeddings_anc, embeddings_anc_pooled = train_util.encode_prompts(pipe, prmpt_batch)
            embeddings_anchor_cache.append(embeddings_anc)
            embeddings_anchor_cache_pooled.append(embeddings_anc_pooled)

        embeddings_anchor_cache = torch.cat(embeddings_anchor_cache, dim=0)
        embeddings_anchor_cache_pooled = torch.cat(embeddings_anchor_cache_pooled, dim=0)

        # Save emb cache
        emb_cache = {
            "embeddings_surrogate_cache": embeddings_surrogate_cache,
            "embeddings_surrogate_cache_pooled": embeddings_surrogate_cache_pooled,
            "embeddings_target_cache": embeddings_target_cache,
            "embeddings_target_cache_pooled": embeddings_target_cache_pooled,
            "embeddings_anchor_cache": embeddings_anchor_cache,
            "embeddings_anchor_cache_pooled": embeddings_anchor_cache_pooled,
            "embeddings_surrogate_sel_base": embeddings_surrogate_sel_base,
            "embeddings_surrogate_sel_base_pooled": embeddings_surrogate_sel_base_pooled,
            "embeddings_target_sel_base": embeddings_target_sel_base,
            "embeddings_target_sel_base_pooled": embeddings_target_sel_base_pooled,
            "embeddings_anchor_sel_base": embeddings_anchor_sel_base,
            "embeddings_anchor_sel_base_pooled": embeddings_anchor_sel_base_pooled,
            "prmpt_scripts_sur": prompt_in_scripts_neutral,
            "prmpt_scripts_tar": prmpt_scripts_tar,
            "prmpt_scripts_anc": prmpt_scripts_anc,
        }


        os.makedirs(emb_cache_path, exist_ok=True)
        torch.save(emb_cache, f"{emb_cache_path}/{emb_cache_fn}")

    return emb_cache




@torch.no_grad()
def get_registered_buffer(args, module_name_list_all, org_modules_all, st_timestep, \
                        end_timestep, n_avail_tokens, prompts, embeddings, embeddings_pooled, embedding_uncond, \
                        pipe, device, register_buffer_path, register_buffer_fn, register_func, **kwargs):

    registered_buffer = dict()
    hooks = []

    registered_buffer, hooks = globals()[register_func](args, module_name_list_all, org_modules_all,\
                            registered_buffer, hooks, \
                            st_timestep, end_timestep, n_avail_tokens, **kwargs)

    embs_batchsize = 1
    embs_batch = []
    pooled_batch = []
    prompts_batch = []
    len_embs_batch = embeddings.size(0)

    os.makedirs(register_buffer_path, exist_ok=True)

    if os.path.isfile(f"{register_buffer_path}/{register_buffer_fn}"):
        print(f"load precomputed registered_buffer for original models ... {register_buffer_path}/{register_buffer_fn}")
        registered_buffer = torch.load(f"{register_buffer_path}/{register_buffer_fn}", map_location=torch.device(device))

    else:
        print(f"compute registered_buffer for original models ... {register_buffer_path}/{register_buffer_fn}")
        for batch_idx in range(int(math.ceil(float(len_embs_batch)/embs_batchsize))):
            if embs_batchsize*(batch_idx+1) <= len_embs_batch:
                embs_batch.append(embeddings[embs_batchsize*batch_idx:embs_batchsize*(batch_idx+1)])
                pooled_batch.append(embeddings_pooled[embs_batchsize*batch_idx:embs_batchsize*(batch_idx+1)])
                prompts_batch.append(prompts[embs_batchsize*batch_idx:embs_batchsize*(batch_idx+1)])
                
            else:
                embs_batch.append(embeddings[embs_batchsize*batch_idx:])
                pooled_batch.append(embeddings_pooled[embs_batchsize*batch_idx:])
                prompts_batch.append(prompts[embs_batchsize*batch_idx:])

        for step, (embs, pooled, prompts) in enumerate(zip(embs_batch, pooled_batch, prompts_batch)):
            if step % 10 == 0:
                print(f"{step}/{len(embs_batch)}")

            for seed in range(args.n_generation_per_concept):

                for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
                    for n in module_name_list:
                        if "seed" in registered_buffer[find_module_name][n].keys():
                            registered_buffer[find_module_name][n]["seed"] = seed

                if len(embs.size())==4:
                    B,C,T,D = embs.size()
                    embs = embs.reshape(B*C,T,D)
                    B,C,D = pooled.size()
                    pooled = pooled.reshape(B*C,D)

                if "save_path" in kwargs.keys():
                    save_path = f"{kwargs['save_path']}/seed_{seed}"
                    os.makedirs(f"{save_path}", exist_ok=True)
                    save_path = f"{save_path}/image.png"

                else:
                    save_path = "./sampled.png"

                seed_everything(seed)

                train_util_SD3.embedding2img_sd3(embs, pooled, pipe, seed=seed, sample_step=end_timestep, device=device, save_path=save_path)
                # train_util.embedding2img(embs, "", pipe, seed=seed, \
                #                     uncond_embeddings=embedding_uncond, end_timestep=end_timestep)

                for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
                    for n in module_name_list:
                        registered_buffer[find_module_name][n]["t"] = 0


            
                # breakpoint()
        
        save_reject_funcs = ["register_norm_buffer_save_activation_sel", \
                            "register_sum_buffer_avg_spatial_rel_debug", \
                            "register_sum_buffer_avg_spatial_rel_debug_sel"]

        if not register_func in save_reject_funcs:
            torch.save(registered_buffer, f"{register_buffer_path}/{register_buffer_fn}")

    for hook in hooks:
        hook.remove()

    return registered_buffer
