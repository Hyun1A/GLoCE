# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
# - https://github.com/Con6924/SPM


import os
from copy import deepcopy
import math
from typing import Optional, List
import numpy as np
from src.models.merge_gloce import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

import hashlib


class ParamModule(nn.Module):
    def __init__(self, size):
        super(ParamModule, self).__init__()
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x * self.weight

    def __repr__(self):
        return f"ParameterModule(param_shape={tuple(self.weight.shape)})"

    



##################################################################################
##################################################################################




class SimpleSelectorOutProp(nn.Module):
    def __init__(self, gate_rank, d_model, dropout=0.25, n_concepts=1, is_last_layer=False):
        super().__init__()        
        
        self.d_model = d_model
        self.gate_rank = gate_rank
        self.n_concepts = n_concepts
        self.temperature = 1.0


        self.select_weight = ParamModule((n_concepts, d_model, gate_rank))
        nn.init.kaiming_uniform_(self.select_weight.weight, a=math.sqrt(5))
        self.select_weight.weight.data = self.select_weight.weight.data / (d_model**2)    

        
        self.select_mean_diff = ParamModule((n_concepts, d_model))
        
        nn.init.kaiming_uniform_(self.select_mean_diff.weight, a=math.sqrt(5))
        self.select_mean_diff.weight.data = self.select_mean_diff.weight.data / (d_model**2)    
        

        self.register_buffer("imp_center", torch.zeros(n_concepts))
        self.register_buffer("imp_slope", torch.zeros(n_concepts))

        self.dropout = nn.Dropout(dropout)
        self.is_last_layer = is_last_layer

        
    def forward(self, x, mask=None):
        ## x: (B,T,D)
        
        x = x.unsqueeze(1)
        x_diff = x - self.select_mean_diff.weight.unsqueeze(0).unsqueeze(2) # BxNxTxD
        x_diff_norm =  x_diff / x_diff.norm(dim=3, keepdim=True)

        Vh_gate = self.select_weight.weight # (N,D,1)        
        cont = torch.einsum("nds,bntd->bnts", Vh_gate,x_diff_norm)**2
        
        select_scale = torch.sigmoid( self.imp_slope.unsqueeze(0).unsqueeze(-1)*( \
                    cont.sum(dim=-1) - self.imp_center.unsqueeze(0).unsqueeze(-1)) ) # BN


        select_scale, select_idx = select_scale.max(dim=1, keepdim=True) #


        return select_idx, select_scale

    def reset_select_cache(self):
        self.sel_prop.select_scale_cache = None
        self.sel_prop.prop_num = 0




class GLoCELayerOutProp(nn.Module):
    
    def __init__(
        self,
        find_name,
        gloce_name,
        gloce_org_name,
        org_module: nn.Module,
        multiplier=1.0,
        alpha=1,
        gate_rank=1,
        update_rank=4,
        degen_rank=4,
        n_concepts=1,
        last_layer_name="",
        use_update=True,
        use_degen=True,
        use_bias=True,
        use_gate=True,
        st_step=10,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.find_name = find_name
        self.gloce_name = gloce_name
        self.gloce_org_name = gloce_org_name
        self.eta = 1.0
        self.st_step = st_step
        self.n_step = 51

        self.use_update = use_update
        self.use_degen = use_degen
        self.use_bias = use_bias
        self.use_gate = use_gate


        

        if org_module.__class__.__name__ == "Linear":
 
            out_dim = org_module.out_features

            self.lora_update = ParamModule((n_concepts,out_dim,degen_rank))
            self.lora_degen = ParamModule((n_concepts,out_dim,degen_rank))

            self.bias = ParamModule((1,n_concepts,out_dim))
            self.debias = ParamModule((1,n_concepts,out_dim))


        # same as microsoft's
        nn.init.zeros_(self.lora_update.weight)    
        nn.init.zeros_(self.lora_degen.weight)    
            
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = update_rank if alpha is None or alpha == 0 else alpha
        self.scale = alpha / update_rank
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying


        is_last_layer = True if gloce_org_name == last_layer_name else False


        self.selector = SimpleSelectorOutProp(gate_rank=gate_rank, d_model=out_dim,\
                                            dropout=0.25, n_concepts=n_concepts, is_last_layer=is_last_layer)
        
        self.use_prompt_tuning = False    
        self.t_counter = 0



    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # x.shape: (B, 77, 768)
        x = self.org_forward(x)
        self.t_counter +=1
        if self.use_prompt_tuning:    
            if self.t_counter > self.st_step:
                select_idx, select_scale = self.selector(x) # (BxT)
                                
                debias = self.debias.weight.squeeze(0)[select_idx.squeeze(1)]
                x_debias = x-debias # BxTxD
    
                update_mat_sel = self.lora_update.weight[select_idx.squeeze(1)]
                degen_mat_sel = self.lora_degen.weight[select_idx.squeeze(1)]
                                    
                mod_x = torch.einsum("btdh,btd->bth", update_mat_sel, x_debias)
                degen_up = torch.einsum("btdh,bth->btd", degen_mat_sel, mod_x)

                bias = self.bias.weight.squeeze(0)[select_idx.squeeze(1)]
                mod_x_bias = self.eta * (bias + degen_up) # BxNxTxD

                if not self.use_gate:
                    select_scale = torch.ones_like(select_scale).to(x.device)
                
                if self.t_counter == self.n_step:
                    self.t_counter = 0


                return (1-select_scale.permute(0,2,1))*x + select_scale.permute(0,2,1)*mod_x_bias  #                
                
             
            else:
                return x
             
        else:
            return x 



class GLoCENetworkOutProp(nn.Module):
    TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    GLoCE_PREFIX = "lora_gloce"   # aligning with SD webui usage
    DEFAULT_TARGET_REPLACE = TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        diffusion_model,
        text_encoder: CLIPTextModel,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module = GLoCELayerOutProp,
        module_kwargs = None,
        delta=1e-5,
        gate_rank=4,
        update_rank=4,
        degen_rank=4,
        n_concepts=1,
        org_modules_all=None,
        module_name_list_all=None,  
        find_module_names = None,
        last_layer = "",
        st_step = 10,
    ) -> None:
        
        super().__init__()
        
        self.n_concepts = n_concepts
        
        self.multiplier = multiplier
        self.alpha = alpha
        self.delta = delta 
        
        self.module = module
        self.module_kwargs = module_kwargs or {}
        
        self.gate_rank = gate_rank
        self.update_rank = update_rank
        self.degen_rank = degen_rank

        self.find_module_names = find_module_names
        self.org_modules_all=org_modules_all
        self.module_name_list_all=module_name_list_all
        self.last_layer = last_layer
        self.st_step = st_step

        ####################################################
        ####################################################

        self.gloce_layers = self.create_modules(
            GLoCENetworkOutProp.GLoCE_PREFIX,
            self.multiplier,
        )

        print(f"Create GLoCE for U-Net: {len(self.gloce_layers)} modules.")

        gloce_names = set()
        for gloce_layer in self.gloce_layers:
            assert (
                gloce_layer.gloce_name not in gloce_names
            ), f"duplicated GLoCE layer name: {gloce_layer.gloce_name}. {gloce_names}"
            gloce_names.add(gloce_layer.gloce_name)

        ############### Add: printing modified text encoder module ################
        for gloce_layer in self.gloce_layers:
            gloce_layer.apply_to()
            self.add_module(
                gloce_layer.gloce_name,
                gloce_layer,
            )
        
        del diffusion_model
        

    def load_gloce_lora_models(self, model_paths):
        for layer in self.gloce_layers:
            self.attention.encoder_layer.add_slf_attn(model_paths, layer.gloce_name)
        

    def create_modules(
        self,
        prefix: str,
        multiplier: float,
    ) -> list:
        gloce_layers = []

        for find_name, org_modules in zip(self.find_module_names, self.org_modules_all):
            for module_name, org_module in org_modules.items():
                gloce_org_name = module_name
                gloce_name = prefix + "." + module_name
                gloce_name = gloce_name.replace(".", "_")
                print(f"{gloce_name}")

                gloce_layer = self.module(
                    find_name, gloce_name, gloce_org_name, org_module, multiplier, self.alpha, \
                    gate_rank=self.gate_rank, update_rank=self.update_rank, degen_rank = self.degen_rank, \
                    n_concepts=self.n_concepts,
                    last_layer_name=self.last_layer,
                    st_step=self.st_step,
                    **self.module_kwargs
                )
                gloce_layers.append(gloce_layer)

        return gloce_layers
    

    
    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        
        state_dict = self.state_dict()
        
        state_dict_save = dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict_save[key] = v
                
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict_save, file, metadata)
        else:
            torch.save(state_dict_save, file)


    
    def __enter__(self):
        for gloce_layer in self.gloce_layers:
            gloce_layer.multiplier = 1.0
            gloce_layer.use_prompt_tuning = True

            
    def __exit__(self, exc_type, exc_value, tb):
        for gloce_layer in self.gloce_layers:
            gloce_layer.multiplier = 0
            gloce_layer.use_prompt_tuning = False
                        

##################################################################################
##################################################################################
##################################################################################
##################################################################################    

