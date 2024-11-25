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

import torch
import torch.nn.functional as F

from torch.optim import Optimizer
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin, DiffusionPipeline
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from src.models.model_util import SDXL_TEXT_ENCODER_TYPE
import src.engine.train_util as train_util

from tqdm import tqdm
from PIL import Image


@torch.no_grad()
def register_sum_buffer_avg_spatial(args, module_name_list_all, org_modules_all, registered_buffer, hooks, \
                    st_timestep, end_timestep, n_avail_tokens, **kwargs):
    def get_org_hook(find_m_name, name):
        def forward_hook(module, input, output):
            t = registered_buffer[find_m_name][name]["t"]
            if (t > st_timestep) and (t <= end_timestep):
                B, T, D = output.size()
                

                data = output[B//2:]
                if ("attn2.to_k" in name) or ("attn2.to_v" in name):
                    data_flat = data[0][1:(1+n_avail_tokens)].view((B//2) * n_avail_tokens, -1)
                else:
                    data_flat = data.view((B//2)*T, -1)
                    width, height = int(data.size(1)**0.5), int(data.size(1)**0.5)
                    avg_depth = 4
                    data_spatial = [data.transpose(-1,-2).reshape(B//2, D, width, height)]
                    data_flatten = [data_spatial[-1].view(B//2,D, -1).transpose(-1,-2)]
                    b,t,d = data_flatten[-1].size()
                    data_flatten[-1] = data_flatten[-1].view(b*t,-1)

                    for _ in range(avg_depth-1):
                        data_spatial.append( F.avg_pool2d(data_spatial[-1], kernel_size=2, stride=2) )
                        # if data_spatial[-1].shape[-1] > 16 :
                        #     continue 
                        data_flatten.append(data_spatial[-1].view(B//2,D, -1).transpose(-1,-2))
                        b,t,d = data_flatten[-1].size()
                        data_flatten[-1] = data_flatten[-1].view(b*t,-1)
                    data_flat = torch.cat(data_flatten, dim=0)
                    
                registered_buffer[find_m_name][name]["n_sum_per_forward"] = data_flat.size(0)
                

                data_flat_norm = data_flat / data_flat.norm(dim=-1,keepdim=True)

                if registered_buffer[find_m_name][name]["data"] is None:
                    registered_buffer[find_m_name][name]["data"] = data_flat.T @ data_flat
                    registered_buffer[find_m_name][name]["data_norm"] = data_flat_norm.T @ data_flat_norm
                    registered_buffer[find_m_name][name]["data_mean"] = data_flat.sum(dim=0, keepdim=True)
                else:
                    registered_buffer[find_m_name][name]["data"] += data_flat.T @ data_flat
                    registered_buffer[find_m_name][name]["data_norm"] += data_flat_norm.T @ data_flat_norm
                    registered_buffer[find_m_name][name]["data_mean"] += data_flat.sum(dim=0, keepdim=True)

                registered_buffer[find_m_name][name]["n_forward"] += 1

            registered_buffer[find_m_name][name]["t"] += 1
        return forward_hook

    for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
        registered_buffer[find_module_name] = dict()

        for n in module_name_list:
            registered_buffer[find_module_name][n] = {"t": 0, "data": None, "data_norm": None, "data_mean": None, \
                                                      "n_forward": 0, "n_sum_per_forward": 0}
            hook = org_modules[n].register_forward_hook(get_org_hook(find_module_name, n))
            hooks.append(hook)

    return registered_buffer, hooks
 







@torch.no_grad()
def register_norm_buffer_avg_spatial(args, module_name_list_all, org_modules_all, registered_buffer, hooks, \
                    st_timestep, end_timestep, n_avail_tokens, **kwargs):
    
    def get_org_hook(find_m_name, name):
        def forward_hook(module, input, output):
            rel_gate_dict = kwargs["rel_gate_dict"]
            gate_mean_dict = kwargs["gate_mean_dict"]
            target_mean_dict = kwargs["target_mean_dict"]


            t = registered_buffer[find_m_name][name]["t"]
            if (t > st_timestep) and (t <= end_timestep):
                B, T, D = output.size()
                
                data = output[B//2:]
                if ("attn2.to_k" in name) or ("attn2.to_v" in name):
                    data_flat = data[0][1:(1+n_avail_tokens)].view((B//2) * n_avail_tokens, -1)
                else:
                    data_flat = data.view((B//2)*T, -1)

                    width, height = int(data.size(1)**0.5), int(data.size(1)**0.5)
                    avg_depth = 3
                    data_spatial = [data.transpose(-1,-2).reshape(B//2, D, width, height)]
                    data_flatten = [data_spatial[-1].view(B//2,D, -1).transpose(-1,-2)]
                    b,t,d = data_flatten[-1].size()
                    data_flatten[-1] = data_flatten[-1].view(b*t,-1)

                    # for _ in range(avg_depth-1):
                    #     data_spatial.append( F.avg_pool2d(data_spatial[-1], kernel_size=2, stride=2) )
                    #     # if data_spatial[-1].shape[-1] > 16 :
                    #     #     continue 
                    #     data_flatten.append(data_spatial[-1].view(B//2,D, -1).transpose(-1,-2))
                    #     b,t,d = data_flatten[-1].size()
                    #     data_flatten[-1] = data_flatten[-1].view(b*t,-1)

                    data_flat = torch.cat(data_flatten, dim=0)

                registered_buffer[find_m_name][name]["n_sum_per_forward"] = data_flat.size(0)

                # diff_mean = target_mean_dict[find_m_name][name] - gate_mean_dict[find_m_name][name]
                # diff_mean_norm = diff_mean / diff_mean.norm(dim=1, keepdim=True)

                Vh_rel = rel_gate_dict[find_m_name][name]
                
                diff_data = data_flat - gate_mean_dict[find_m_name][name]
                diff_data_norm = diff_data / diff_data.norm(dim=1, keepdim=True)
                
                data = ((Vh_rel @ diff_data_norm.T)**2).sum(dim=0, keepdim=True)
                
                if registered_buffer[find_m_name][name]["data"] is None:
                    registered_buffer[find_m_name][name]["data"] = data.sum(dim=1)
                    registered_buffer[find_m_name][name]["data_sq"] = (data**2).sum(dim=1)
                    
                    val_max, ind_max = data.sum(dim=0).max(dim=0)  # D --> 1
                    
                    registered_buffer[find_m_name][name]["data_max"] = val_max.clone()
                    registered_buffer[find_m_name][name]["data_max_sq"] = val_max.clone()**2
                    registered_buffer[find_m_name][name]["data_stack"].append( val_max.clone() )

                    registered_buffer[find_m_name][name]["data_maxmax"] = val_max.clone()
                    registered_buffer[find_m_name][name]["data_maxmin"] = val_max.clone()

                else:
                    registered_buffer[find_m_name][name]["data"] += data.sum(dim=1)
                    registered_buffer[find_m_name][name]["data_sq"] += (data**2).sum(dim=1)
                    val_max, ind_max = data.sum(dim=0).max(dim=0)  # D --> 1

                    registered_buffer[find_m_name][name]["data_max"] += val_max.clone()
                    registered_buffer[find_m_name][name]["data_max_sq"] += val_max.clone()**2
                    registered_buffer[find_m_name][name]["data_stack"].append( val_max.clone() )

                    val_cat_max = torch.cat([registered_buffer[find_m_name][name]["data_maxmax"].unsqueeze(0),\
                                            val_max.clone().unsqueeze(0)], dim=0)
                    val_cat_min = torch.cat([registered_buffer[find_m_name][name]["data_maxmin"].unsqueeze(0),\
                                            val_max.clone().unsqueeze(0)], dim=0)
                    val_max_new, _ = val_cat_max.max(dim=0)
                    val_min_new, _ = val_cat_min.min(dim=0)

                    registered_buffer[find_m_name][name]["data_maxmax"] = val_max_new
                    registered_buffer[find_m_name][name]["data_maxmin"] = val_min_new


                registered_buffer[find_m_name][name]["n_forward"] += 1

            registered_buffer[find_m_name][name]["t"] += 1
        return forward_hook

    for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
        registered_buffer[find_module_name] = dict()

        for n in module_name_list:
            registered_buffer[find_module_name][n] = {"t": 0, "data": None, "data_sq": None, \
                                                      "data_max": None, "data_max_sq": None, \
                                                      "data_maxmax": None, "data_max_maxmin": None, \
                                                      "data_stack": [], \
                                                      "n_forward": 0, "n_sum_per_forward": 0}
            hook = org_modules[n].register_forward_hook(get_org_hook(find_module_name, n))
            hooks.append(hook)

    return registered_buffer, hooks

