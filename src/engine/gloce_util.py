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
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from src.models.model_util import SDXL_TEXT_ENCODER_TYPE
import src.engine.train_util as train_util

from .gloce_register_buffer import *



def get_module_name_type(find_module_name):
    if find_module_name == "unet_ca":
        module_type = "Linear"
        module_name = "attn2"

    elif find_module_name == "unet_ca_kv":
        module_type = "Linear"
        module_name = "attn2"

    elif find_module_name == "unet_ca_v":
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


    elif find_module_name == "unet_ca_v":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_v" in n):
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
def prepare_text_embedding_token(args, config, prompts_target, prompts_surround, prompts_update, tokenizer, text_encoder, train_util, DEVICE_CUDA,\
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
        prmpt_temp_sel_base = f"A face of {replace_word}"
    elif replace_word == "explicit":
        prmpt_temp_sel_base = replace_word

    prompt_scripts_list.append( prmpt_temp_sel_base )



    if args.use_emb_cache and os.path.isfile(f"{emb_cache_path}/{emb_cache_fn}"):
        print("load pre-computed text emb cache...")
        emb_cache = torch.load(f"{emb_cache_path}/{emb_cache_fn}", map_location=torch.device(DEVICE_CUDA))
        
    else:
        # Prepare for sel basis simWords
        print("compute text emb cache...")

        

            
            
        #####################################################################
        ##################### compute concept embeddings ####################
        
        # compute target and update concept embeddings
        simWords_target = [prompt.target for prompt in prompts_target]
        prmpt_sel_base_target = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_target] 
        embeddings_target_sel_base = train_util.encode_prompts(tokenizer, text_encoder, prmpt_sel_base_target)

        simWords_update = [prompt.target for prompt in prompts_update]
        prmpt_sel_base_update = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_update] 
        embeddings_update_sel_base = train_util.encode_prompts(tokenizer, text_encoder, prmpt_sel_base_update)            
                
        # Compute and select anchor and surrogate concept embeddings
        simWords_surround = [prompt.target for prompt in prompts_surround]
        for simW_erase in simWords_target:
            simWords_surround = [item.lower() for item in simWords_surround if simW_erase not in item.lower()]
            
        prmpt_sel_base_surround = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_surround] 

        len_surround = len(prmpt_sel_base_surround)
        text_encode_batch = 100
        simWords_surround_batch = [
            prmpt_sel_base_surround[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_surround) / text_encode_batch)))
        ]

        embeddings_surround_sel_base = []
        for simW_batch in simWords_surround_batch:
            emb_surround = train_util.encode_prompts(tokenizer, text_encoder, simW_batch)
            embeddings_surround_sel_base.append(emb_surround)
        embeddings_surround_sel_base = torch.cat(embeddings_surround_sel_base, dim=0)
    

        # Compute similarity
        embeddings_target_norm = embeddings_target_sel_base / embeddings_target_sel_base.norm(2, dim=-1, keepdim=True)
        embeddings_surround_norm = embeddings_surround_sel_base / embeddings_surround_sel_base.norm(2, dim=-1, keepdim=True)

        similarity = torch.einsum("ijk,njk->inj", embeddings_target_norm[:, 1:(1 + n_avail_tokens), :],
                                    embeddings_surround_norm[:, 1:(1 + n_avail_tokens), :]).mean(dim=2)

        similarity = similarity.mean(dim=0)
        val_sorted, ind_sorted = similarity.sort()
        ind_sorted_list = ind_sorted.cpu().numpy().tolist()
        
        
        simWords_anchor = [simWords_surround[sim_idx] for sim_idx in ind_sorted_list[-n_anchor_concepts:]]
        embeddings_anchor_sel_base = embeddings_surround_sel_base[ind_sorted_list[-n_anchor_concepts:]]

        if replace_word == "celeb": # or args.opposite_for_map:        
            simWords_surrogate = [simWords_surround[sim_idx] for sim_idx in ind_sorted_list[:n_anchor_concepts]]
            embeddings_surrogate_sel_base = embeddings_surround_sel_base[ind_sorted_list[:n_anchor_concepts]]  
            
        else:
            simWords_surrogate = [prompts_target[0].neutral]
            prmpt_sel_base_surrogate = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_surrogate] 
            embeddings_surrogate_sel_base = train_util.encode_prompts(tokenizer, text_encoder, prmpt_sel_base_surrogate)        
        
        ##################### compute concept embeddings ####################
        #####################################################################
        
        
        # Prepare for surrogate token cache
        print("compute emb cache...")
        
        embeddings_surrogate_cache = []
        prmpt_scripts_sur = []

        for simWord in simWords_surrogate:
            for prompt_script in prompt_scripts_list:
                pr_in_script_sur = prompt_script.replace(replace_word, simWord)
                pr_in_script_sur = pr_in_script_sur.replace(replace_word.lower(), simWord)
                prmpt_scripts_sur.append(pr_in_script_sur)
                
                
        len_surrogate = len(prmpt_scripts_sur)
        text_encode_batch = 100
        prmpt_scripts_sur_batch = [
            prmpt_scripts_sur[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_surrogate) / text_encode_batch)))
        ]

        
        for prmpt_batch in prmpt_scripts_sur_batch:
            embeddings_sur = train_util.encode_prompts(tokenizer, text_encoder, prmpt_batch)
            embeddings_surrogate_cache.append(embeddings_sur)

        embeddings_surrogate_cache = torch.cat(embeddings_surrogate_cache, dim=0)        
        
                                              
        # Prepare for target token cache
        embeddings_target_cache = []
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
            embeddings_tar = train_util.encode_prompts(tokenizer, text_encoder, prmpt_batch)
            embeddings_target_cache.append(embeddings_tar)

        embeddings_target_cache = torch.cat(embeddings_target_cache, dim=0)


        # Prepare for anchor token cache
        embeddings_anchor_cache = []
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
            embeddings_anc = train_util.encode_prompts(tokenizer, text_encoder, prmpt_batch)
            embeddings_anchor_cache.append(embeddings_anc)

        embeddings_anchor_cache = torch.cat(embeddings_anchor_cache, dim=0)



        # Prepare for update token cache
        embeddings_update_cache = []
        prmpt_scripts_upd = []

        for simWord in simWords_update:
            for prompt_script in prompt_scripts_list:
                pr_in_script_upd = prompt_script.replace(replace_word, simWord)
                # pr_in_script_upd = pr_in_script_upd.replace(replace_word.lower(), simWord)
                prmpt_scripts_upd.append(pr_in_script_upd)

                
        len_update = len(prmpt_scripts_upd)
        text_encode_batch = 100
        prmpt_scripts_upd_batch = [
            prmpt_scripts_upd[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_update) / text_encode_batch)))
        ]
        


        for prmpt_batch in prmpt_scripts_upd_batch:
            embeddings_upd = train_util.encode_prompts(tokenizer, text_encoder, prmpt_batch)
            embeddings_update_cache.append(embeddings_upd)

        embeddings_update_cache = torch.cat(embeddings_update_cache, dim=0)

                                              
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
            "prmpt_scripts_sur": prmpt_scripts_sur,
            "prmpt_scripts_tar": prmpt_scripts_tar,
            "prmpt_scripts_anc": prmpt_scripts_anc,
            "prmpt_scripts_upd": prmpt_scripts_upd,
        }


        os.makedirs(emb_cache_path, exist_ok=True)
        torch.save(emb_cache, f"{emb_cache_path}/{emb_cache_fn}")

    return emb_cache




@torch.no_grad()
def get_registered_buffer(args, module_name_list_all, org_modules_all, st_timestep, \
                        end_timestep, n_avail_tokens, prompts, embeddings, embedding_uncond, \
                        pipe, device, register_buffer_path, register_buffer_fn, register_func, **kwargs):

    registered_buffer = dict()
    hooks = []

    registered_buffer, hooks = globals()[register_func](args, module_name_list_all, org_modules_all,\
                            registered_buffer, hooks, \
                            st_timestep, end_timestep, n_avail_tokens, **kwargs)

    embs_batchsize = 1
    embs_batch = []
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
                prompts_batch.append(prompts[embs_batchsize*batch_idx:embs_batchsize*(batch_idx+1)])
            
            else:
                embs_batch.append(embeddings[embs_batchsize*batch_idx:])
                prompts_batch.append(prompts[embs_batchsize*batch_idx:])

        for step, (embs, prompts) in enumerate(zip(embs_batch, prompts_batch)):
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

                if "save_path" in kwargs.keys():
                    save_path = f"{kwargs['save_path']}/seed_{seed}"
                    os.makedirs(f"{save_path}", exist_ok=True)
                    save_path = f"{save_path}/image.png"

                else:
                    save_path = "./test2.png"

                train_util.embedding2img(embs, "", pipe, seed=seed, \
                                    uncond_embeddings=embedding_uncond, end_timestep=end_timestep, save_path=save_path)

                for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
                    for n in module_name_list:
                        registered_buffer[find_module_name][n]["t"] = 0

        for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
            for n in module_name_list:
                registered_buffer[find_module_name][n]["t"] = 0

        if register_func != "register_norm_buffer_save_activation_sel":
            torch.save(registered_buffer, f"{register_buffer_path}/{register_buffer_fn}")

    for hook in hooks:
        hook.remove()

    return registered_buffer
