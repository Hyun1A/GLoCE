# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
# - https://github.com/Con6924/SPM

import argparse
from pathlib import Path
import gc
from copy import deepcopy
import pandas as pd
import random

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import os, sys
import numpy as np

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

from src.models.merge_gloce import *
from src.models.gloce import(
    GLoCELayerOutProp,
    GLoCENetworkOutProp,
)


import src.engine.train_util as train_util
import src.engine.gloce_util as gloce_util
from src.models import model_util
from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.configs.config import RootConfig
from src.configs.prompt import PromptEmbedsCache, PromptEmbedsPair, PromptSettings


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
DEVICE_CUDA = torch.device("cuda")


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


def train(
    config: RootConfig,
    prompts_target: list[PromptSettings],
    prompts_anchor: list[PromptSettings],
    prompts_update: list[PromptSettings],
    args,
):
    
    ########################################################   
    ################### Setup for GLoCE #####################

    n_target_concepts = args.n_target_concepts
    tar_concept_idx = args.tar_concept_idx
    n_anchor_concepts = args.n_anchor_concepts
    st_timestep = args.st_timestep
    end_timestep = args.end_timestep
    n_avail_tokens = args.n_tokens
    eta = args.eta
    lamb = args.lamb
    update_rank = args.update_rank
    gate_rank = args.gate_rank
    degen_rank = args.degen_rank

    prompts_target = prompts_target[tar_concept_idx:tar_concept_idx+n_target_concepts]

    targets = [prompt.target for prompt in prompts_target]
    anchors = [prompt.target for prompt in prompts_anchor]
    surrogate = [prompts_target[0].neutral]
    updates = [prompt.target for prompt in prompts_update]


    targets_fn = [prompt.target.replace(" ", "_") for prompt in prompts_target]
    anchors_fn = [prompt.target.replace(" ", "_") for prompt in prompts_anchor]

    save_path = f"{args.save_path}/{targets[0].replace(' ', '_')}"     
    param_cache_path = args.param_cache_path 
    emb_cache_path = f"{args.emb_cache_path}/{targets[0].replace(' ', '_')}"
    register_buffer_path = f"{args.buffer_path}/{targets[0].replace(' ', '_')}"
    emb_cache_fn = args.emb_cache_fn

    if os.path.isfile(f"{save_path}/ckpt.safetensors"):
        print(f"ckpt for {tar_concept_idx}-{targets[0]} exists")
        return

    
    ################### Setup for GLoCE #####################
    ########################################################   
    

        
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts_target]),
        "config": config.json(),
    }
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts_target]),
        "rank": str(config.network.rank),
        "alpha": str(config.network.alpha),
    }

    if config.logging.verbose:
        print(metadata)

    weight_dtype = config_pkg.parse_precision(config.train.precision)
    save_weight_dtype = config_pkg.parse_precision(config.train.precision)
        

    (
        tokenizer, 
        text_encoder, 
        unet, 
        noise_scheduler, 
        pipe
    ) = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
        device= DEVICE_CUDA
    )


    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()
    
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    pipe.safety_checker = None

    
    ####################################################################################
    ############################## register org modules ################################              
    module_types = []
    module_names = []
    org_modules_all = []
    module_name_list_all = []
    param_vh_cache_dict_all = []
    param_s_cache_dict_all = []
    
    for find_module_name in args.find_module_name:
        module_name, module_type = gloce_util.get_module_name_type(find_module_name)            
        org_modules, module_name_list = gloce_util.get_modules_list(unet, text_encoder, \
                                                    find_module_name, module_name, module_type)
        param_vh_cache_dict, param_s_cache_dict = gloce_util.load_model_sv_cache(find_module_name, \
                                                    param_cache_path, DEVICE_CUDA, org_modules)

        module_names.append(module_name)
        module_types.append(module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)
        param_vh_cache_dict_all.append(param_vh_cache_dict)
        param_s_cache_dict_all.append(param_s_cache_dict)
    ############################## register org modules ################################              
    ####################################################################################
    
    ########################################################        
    ################### Prepare network ####################



    network = GLoCENetworkOutProp(
        unet,
        text_encoder,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=GLoCELayerOutProp,
        gate_rank=gate_rank,
        update_rank=update_rank,
        degen_rank=degen_rank,
        n_concepts=1,
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names = args.find_module_name,
        last_layer=args.last_layer,
    ).to(DEVICE_CUDA, dtype=weight_dtype)    

    print()
    print("gate rank of netowrk:" , config.network.init_size)
    print()

    network.eval()    
    
    with torch.no_grad():
        embedding_unconditional = train_util.encode_prompts(tokenizer, text_encoder, [""])
    
    network_modules = dict()
    for name, module in network.named_modules():
        if "GLoCELayer" in module.__class__.__name__:
            network_modules[name] = module

    unet_modules = dict()
    for name, module in unet.named_modules():
        name = "_".join(name.split("."))
        name = "lora_unet_" + name

        for network_name in network_modules.keys():
            if name == network_name:
                unet_modules[name] = module   
    ################### Prepare network ####################
    ########################################################   


    ####################################################################
    ############### Prepare for text embedding token ###################
    
    emb_cache = gloce_util.prepare_text_embedding_token(args, config, prompts_target, prompts_anchor, prompts_update, \
                                        tokenizer, text_encoder, train_util, DEVICE_CUDA, 
                                        emb_cache_path, emb_cache_fn,
                                        n_avail_tokens=n_avail_tokens,
                                        n_anchor_concepts=n_anchor_concepts)

    embeddings_surrogate_sel_base = emb_cache["embeddings_surrogate_sel_base"]
    embeddings_target_sel_base = emb_cache["embeddings_target_sel_base"]
    embeddings_anchor_sel_base = emb_cache["embeddings_anchor_sel_base"]
    embeddings_update_sel_base = emb_cache["embeddings_update_sel_base"]

    embeddings_surrogate_cache = emb_cache["embeddings_surrogate_cache"]
    embeddings_target_cache = emb_cache["embeddings_target_cache"]
    embeddings_anchor_cache = emb_cache["embeddings_anchor_cache"]
    embeddings_update_cache = emb_cache["embeddings_update_cache"]

    prmpt_scripts_sur = emb_cache["prmpt_scripts_sur"]
    prmpt_scripts_tar = emb_cache["prmpt_scripts_tar"]
    prmpt_scripts_anc = emb_cache["prmpt_scripts_anc"]
    prmpt_scripts_upd = emb_cache["prmpt_scripts_upd"]    
    
    prompt_scripts_path = config.scripts_file
    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list = prompt_scripts_df['prompt'].to_list()
    len_prmpts_list = len(prompt_scripts_list) + 1

    use_prompt = ("unet_ca_v" in args.find_module_name) or ("unet_ca_outv" in args.find_module_name)
    if config.replace_word == "artist" and use_prompt : 
        embeddings_surrogate_sel_base = embeddings_surrogate_cache
        embeddings_target_sel_base = embeddings_target_cache
        embeddings_anchor_sel_base = embeddings_anchor_cache
        embeddings_update_sel_base = embeddings_update_cache

        surrogate = prmpt_scripts_sur
        targets = prmpt_scripts_tar
        anchors = prmpt_scripts_anc
        updates = prmpt_scripts_upd

    
    target_selected = prmpt_scripts_tar[len_prmpts_list-1::len_prmpts_list]
    print("target concept:", target_selected)
    
    anchor_selected = prmpt_scripts_anc[len_prmpts_list-1::len_prmpts_list]
    print("anchor concept:", anchor_selected)

    surrogate_selected = prmpt_scripts_sur[len_prmpts_list-1::len_prmpts_list]
    print("surrogate concept:", surrogate_selected)
        
    neutral_selected = prmpt_scripts_upd[len_prmpts_list-1::len_prmpts_list]
    print("neutral concept:", neutral_selected)



    
    ############### Prepare for text embedding token ###################
    ####################################################################

    ###########################################################################################   
    ################# Compute register buffer for surrogate concept for erasing #################

    register_buffer_fn = "stacked_surrogate.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_surrogate = gloce_util.get_registered_buffer(args, module_name_list_all, \
                            org_modules_all, st_timestep, end_timestep, n_avail_tokens, \
                            surrogate, embeddings_surrogate_sel_base, embedding_unconditional, \
                            pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func)
    
    ################# Compute register buffer for surrogate concept for erasing #################
    ###########################################################################################   



    ##############################################################################################   
    #################### Compute principal components for surrogate concept ######################

    Vh_sur_dict = dict()
    surrogate_mean_dict = dict()
    for find_name in args.find_module_name:
        Vh_sur_dict[find_name] = dict()
        surrogate_mean_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:        
        n_forward = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_per_forward = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]['n_sum_per_forward']
        n_sum = n_forward*n_sum_per_forward

        stacked_buffer_surrogate = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]['data'] / n_sum
        stacked_buffer_surrogate_mean = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]["data_mean"] / n_sum
        stacked_buffer_surrogate_cov = stacked_buffer_surrogate - stacked_buffer_surrogate_mean.T @ stacked_buffer_surrogate_mean
        

        _,S_sur,Vh_sur = torch.linalg.svd(stacked_buffer_surrogate_cov, full_matrices=False)
        Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_sur
        surrogate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_surrogate_mean

        gloce_module.lora_degen.weight.data = Vh_sur[:degen_rank].T.contiguous()
        gloce_module.bias.weight.data = stacked_buffer_surrogate_mean.unsqueeze(0).clone().contiguous()  

    #################### Compute principal components for surrogate concept ######################
    ##############################################################################################   



    ###########################################################################################   
    ################# Compute registder buffer for target concept for erasing #################

    register_buffer_fn = "stacked_target.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_target = gloce_util.get_registered_buffer(args, module_name_list_all, \
                            org_modules_all, st_timestep, end_timestep, n_avail_tokens, \
                            targets, embeddings_target_sel_base, embedding_unconditional, \
                            pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func)

    ################# Compute registder buffer for target concept for erasing #################
    ###########################################################################################   




    ###########################################################################################   
    #################### Compute principal components for target concept ######################

    target_mean_dict = dict()
    target_cov_dict = dict()
    Vh_tar_dict = dict()
    for find_name in args.find_module_name:
        target_mean_dict[find_name] = dict()
        Vh_tar_dict[find_name] = dict()
        target_cov_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:        
        n_forward = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_per_forward = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['n_sum_per_forward']
        n_sum = n_forward*n_sum_per_forward

        stacked_buffer_target_mean = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data_mean'] / n_sum
        stacked_buffer_target = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data'] / n_sum
        stacked_buffer_target_cov = stacked_buffer_target - stacked_buffer_target_mean.T @ stacked_buffer_target_mean

        _,S_tar,Vh_tar = torch.linalg.svd(stacked_buffer_target_cov, full_matrices=False)
        Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_tar[:update_rank]
        target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_mean
        target_cov_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_cov
        
    for gloce_module in network.gloce_layers:   
        Vh_upd = Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name][:update_rank]
        target_mean = target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name].squeeze(0)
        dim_emb = Vh_upd.size(1)

        Vh_sur = Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name][:degen_rank] # hxd        
        gloce_module.lora_update.weight.data = ( Vh_sur@(torch.eye(dim_emb).to(DEVICE_CUDA)-Vh_upd.T@Vh_upd) ).T.contiguous()
        gloce_module.debias.weight.data = target_mean.unsqueeze(0).unsqueeze(0).clone().contiguous()  
    #################### Compute principal components for target concept ######################
    ###########################################################################################   




    
    
    ###########################################################################################   
    #################### Compute register buffer for surrogate for gate #######################

    register_buffer_fn = "stacked_gate.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_gate = gloce_util.get_registered_buffer(args, module_name_list_all, \
                            org_modules_all, st_timestep, end_timestep, n_avail_tokens, \
                            updates, embeddings_update_sel_base, embedding_unconditional, \
                            pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func)
    
    #################### Compute register buffer for surrogate for gate #######################
    ###########################################################################################   
    
    ##############################################################################################   
    #################### Compute principal components of surrogate for gate ######################

    Vh_gate_dict = dict()
    gate_mean_dict = dict()
    rel_gate_dict = dict()
    for find_name in args.find_module_name:
        Vh_gate_dict[find_name] = dict()
        gate_mean_dict[find_name] = dict()
        rel_gate_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:        
        n_forward = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_per_forward = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]['n_sum_per_forward']
        n_sum = n_forward*n_sum_per_forward

        stacked_buffer_gate = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]['data'] / n_sum
        stacked_buffer_gate_mean = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]["data_mean"] / n_sum
        stacked_buffer_gate_cov = stacked_buffer_gate - stacked_buffer_gate_mean.T @ stacked_buffer_gate_mean
        
        
        
        stacked_buffer_rel_mean = target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] \
                                    - stacked_buffer_gate_mean
        stacked_buffer_rel_cov = target_cov_dict[gloce_module.find_name][gloce_module.gloce_org_name] \
                                    + stacked_buffer_rel_mean.T @ stacked_buffer_rel_mean
                
        _,S_tar,Vh_gate = torch.linalg.svd(stacked_buffer_rel_cov, full_matrices=False)
        rel_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_gate[:gate_rank]
        gate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_gate_mean
        
        
    #################### Compute principal components of surrogate for gate ######################
    ##############################################################################################   

    ###########################################################################################   
    ############## Compute registder buffer for discriminative basis for erasing ##############

    register_buffer_fn = "norm_target.pt"
    register_func = "register_norm_buffer_avg_spatial"
    
    buffer_norm_basis_target = gloce_util.get_registered_buffer(args, module_name_list_all, \
                            org_modules_all, st_timestep, end_timestep, n_avail_tokens, \
                            targets, embeddings_target_sel_base, embedding_unconditional, \
                            pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func, \
                            rel_gate_dict=rel_gate_dict, \
                            target_mean_dict=target_mean_dict, gate_mean_dict=gate_mean_dict)


    register_buffer_fn = "norm_anchor.pt"
    register_func = "register_norm_buffer_avg_spatial"

    buffer_norm_basis_anchor = gloce_util.get_registered_buffer(args, module_name_list_all, \
                            org_modules_all, st_timestep, end_timestep, n_avail_tokens, \
                            anchors, embeddings_anchor_sel_base, embedding_unconditional, \
                            pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func, \
                            rel_gate_dict=rel_gate_dict, \
                            target_mean_dict=target_mean_dict, gate_mean_dict=gate_mean_dict)

    ############## Compute registder buffer for discriminative basis for erasing ##############
    ###########################################################################################   



    ######################################################################
    ############## Compute discriminative basis for erasing ##############
 
    for gloce_module in network.gloce_layers:        
        n_forward_tar = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_forward_anc = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_tar = n_forward_tar
        n_sum_anc = n_forward_anc

        importance_tgt = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data_max'] / n_sum_tar
        importance_anc = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]['data_max'] / n_sum_anc



        importance_tgt_stack = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data_stack']
        importance_anc_stack = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]['data_stack']
        importance_tgt_stack = torch.cat([imp.unsqueeze(0) for imp in importance_tgt_stack], dim=0)
        importance_anc_stack = torch.cat([imp.unsqueeze(0) for imp in importance_anc_stack], dim=0)


        print(gloce_module.gloce_org_name)
        print(f"Relative importance", ( importance_tgt / importance_anc ).item())

        ########### Determine parameters in logistic function ############

        x_center = importance_anc_stack.mean() + args.thresh*importance_anc_stack.std()     
        tol1 = 0.1*args.thresh*importance_anc_stack.std()  

        x_left = x_center - tol1
        x_right = x_center + tol1

        c_left = torch.tensor([0.01]).to(DEVICE_CUDA)
        c_right = torch.tensor([0.99]).to(DEVICE_CUDA)
        
        C_left = torch.log(1/(1/c_left - 1))
        C_right = torch.log(1/(1/c_right - 1))

        imp_center = ( (C_left/C_right) * x_right - x_left ) / ( (C_left/C_right) - 1 )
        imp_slope = C_left * (1/(x_left-imp_center))

        print(f"{importance_anc_stack.max().item():10.5f}, {imp_center.item():10.5f}, {importance_tgt_stack.min().item():10.5f}, {importance_tgt_stack.max().item():10.5f}")
        ########### Determine parameters in logistic function ############

        
        Vh_gate = rel_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name]
        gate_mean = gate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name]


        # NxD
        gloce_module.selector.select_weight.weight.data = Vh_gate.T.unsqueeze(0).clone().contiguous()
        gloce_module.selector.select_mean_diff.weight.data = gate_mean.clone().contiguous()

        gloce_module.selector.imp_center = imp_center
        gloce_module.selector.imp_slope = imp_slope

        print()

    ############## Compute discriminative basis for erasing ##############
    ######################################################################   
    

    print("saving gloce parameters...")
    save_path = Path(f"{save_path}")            
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"ckpt.safetensors",
        dtype=save_weight_dtype,
        metadata=model_metadata,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_pkg.load_config_from_yaml(config_file)
        
    prompts_target = prompt_pkg.load_prompts_from_yaml(config.prompts_file_target)
    prompts_anchor = prompt_pkg.load_prompts_from_yaml(config.prompts_file_anchor)
    prompts_update = prompt_pkg.load_prompts_from_yaml(config.prompts_file_update)
    
    if args.gate_rank != -1:
        config.network.init_size = args.gate_rank
        config.network.hidden_size = args.gate_rank
        config.network.continual_rank = args.gate_rank
            
    if args.update_rank != -1:
        config.network.rank = args.update_rank     

    base_logging_prompts = config.logging.prompts
    
    for p_idx, p in enumerate(prompts_target):
        config.logging.prompts = [prompt.replace('[target]', p.target) if '[target]' in prompt else prompt for prompt in base_logging_prompts]
    
    args.find_module_name = args.find_module_name.split(",")
    if args.find_module_name.__class__ == str:
        args.find_module_name = [args.find_module_name]

    seed_everything(config.train.train_seed)        
    train(config, prompts_target, prompts_anchor, prompts_update, args)
    






    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", required=True, help="Config file for training.")
    parser.add_argument("--st_prompt_idx", type=int, default=-1)
    parser.add_argument("--end_prompt_idx", type=int, default=-1)
    parser.add_argument("--update_rank", type=int, default=-1)
    parser.add_argument("--degen_rank", type=int, default=-1)
    parser.add_argument("--gate_rank", type=int, default=-1)
    parser.add_argument("--n_tokens", type=int, default=-1)
    parser.add_argument("--eta", type=float, default=-1)
    parser.add_argument("--lamb", type=float, default=-1)
    parser.add_argument("--lamb2", type=float, default=-1)
    parser.add_argument("--p_val", type=float, default=-1)
    parser.add_argument("--find_module_name", type=str, default="unet_ca")


    parser.add_argument('--n_target_concepts', type=int, default=1, help="Number of target concepts")
    parser.add_argument('--n_anchor_concepts', type=int, default=5, help="Number of anchor concepts")
    parser.add_argument('--tar_concept_idx', type=int, default=0, help="Target concept index")
    parser.add_argument('--st_timestep', type=int, default=10, help="Start timestep")
    parser.add_argument('--end_timestep', type=int, default=20, help="End timestep")
    parser.add_argument('--n_generation_per_concept', type=int, default=3, help="End timestep")
    parser.add_argument('--sel_basis_buffer_fn', action='store_true', help="Select basis buffer function")
    parser.add_argument('--param_cache_path', type=str, default="./importance_cache/org_comps/sd_v1.4", help="Path to parameter cache")
    parser.add_argument('--emb_cache_path', type=str, default="./importance_cache/text_embs/sd_v1.4", help="Path to embedding cache")
    parser.add_argument('--emb_cache_fn', type=str, default="text_emb_cache_w_sel_base_chris_evans_anchor5.pt", help="Embedding cache file name")
    parser.add_argument("--buffer_path", type=str, default="./importance_cache/buffers")
    parser.add_argument("--use_emb_cache", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--last_layer", type=str, default="")
    parser.add_argument("--opposite_for_map", type=bool, default=False)
    parser.add_argument("--thresh", type=float, default=1.5)




    args = parser.parse_args()
        
    main(args)
