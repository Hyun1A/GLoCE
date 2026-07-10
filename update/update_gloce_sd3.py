# SD3 port of the official GLoCE update script (github.com/Hyun1A/GLoCE, update/update_gloce.py)
# Uses StableDiffusion3Pipeline and injects GLoCE into attn.to_out.0 of each JointTransformerBlock.

import argparse
from pathlib import Path
import gc
import random

import numpy as np
import torch
import os, sys

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

from src.models.merge_gloce import *
from src.models.gloce_sd3 import (
    GLoCELayerOutProp,
    GLoCENetworkOutProp,
)

import src.engine.train_util_sd3 as train_util_SD3
import src.engine.nice_util_sd3 as nice_util_SD3
from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.configs.config import RootConfig
from src.configs.prompt import PromptSettings

from diffusers import StableDiffusion3Pipeline

DEVICE_CUDA = torch.device("cuda:0")


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
    n_target_concepts = args.n_target_concepts
    tar_concept_idx = args.tar_concept_idx
    n_anchor_concepts = args.n_anchor_concepts
    st_timestep = args.st_timestep
    end_timestep = args.end_timestep
    n_avail_tokens = args.n_tokens
    update_rank = args.update_rank
    gate_rank = args.gate_rank
    degen_rank = args.degen_rank

    prompts_target = prompts_target[tar_concept_idx:tar_concept_idx+n_target_concepts]

    targets = [prompt.target for prompt in prompts_target]
    anchors = [prompt.target for prompt in prompts_anchor]
    surrogate = [prompts_target[0].neutral]
    updates = [prompt.target for prompt in prompts_update]

    save_path = f"{args.save_path}/{targets[0].replace(' ', '_')}"
    emb_cache_path = f"{args.emb_cache_path}/{targets[0].replace(' ', '_')}"
    register_buffer_path = f"{args.buffer_path}/{targets[0].replace(' ', '_')}"
    emb_cache_fn = args.emb_cache_fn

    if os.path.isfile(f"{save_path}/ckpt.safetensors"):
        print(f"ckpt for {tar_concept_idx}-{targets[0]} exists")
        return

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts_target]),
        "config": config.json(),
    }
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts_target]),
        "rank": str(config.network.rank),
        "alpha": str(config.network.alpha),
    }

    weight_dtype = config_pkg.parse_precision(config.train.precision)
    save_weight_dtype = config_pkg.parse_precision(config.train.precision)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32,
        text_encoder_3=None, tokenizer_3=None).to(DEVICE_CUDA)
    pipe.transformer.eval()
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()
    pipe.safety_checker = None

    ############################## register org modules ################################
    org_modules_all = []
    module_name_list_all = []

    for find_module_name in args.find_module_name:
        module_name, module_type = nice_util_SD3.get_module_name_type(find_module_name)
        org_modules, module_name_list = nice_util_SD3.get_modules_list(
            pipe.transformer, pipe.text_encoder, find_module_name, module_name, module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)

    ################### Prepare network ####################
    network = GLoCENetworkOutProp(
        pipe.transformer,
        pipe.text_encoder,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=GLoCELayerOutProp,
        gate_rank=gate_rank,
        update_rank=update_rank,
        degen_rank=degen_rank,
        n_concepts=1,
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names=args.find_module_name,
        last_layer=args.last_layer,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    network.eval()

    with torch.no_grad():
        embedding_unconditional, embedding_uncond_pooled = train_util_SD3.encode_prompts(pipe, [""])

    ############### Prepare for text embedding token ###################
    emb_cache = nice_util_SD3.prepare_text_embedding_token(
        args, config, prompts_target, prompts_anchor, prompts_update,
        pipe, train_util_SD3, DEVICE_CUDA,
        emb_cache_path, emb_cache_fn,
        n_avail_tokens=n_avail_tokens,
        n_anchor_concepts=n_anchor_concepts)

    embeddings_surrogate_sel_base = emb_cache["embeddings_surrogate_sel_base"]
    embeddings_target_sel_base = emb_cache["embeddings_target_sel_base"]
    embeddings_anchor_sel_base = emb_cache["embeddings_anchor_sel_base"]
    embeddings_update_sel_base = emb_cache["embeddings_update_sel_base"]
    embeddings_surrogate_sel_base_pooled = emb_cache["embeddings_surrogate_sel_base_pooled"]
    embeddings_target_sel_base_pooled = emb_cache["embeddings_target_sel_base_pooled"]
    embeddings_anchor_sel_base_pooled = emb_cache["embeddings_anchor_sel_base_pooled"]
    embeddings_update_sel_base_pooled = emb_cache["embeddings_update_sel_base_pooled"]

    print("target concept:", targets)
    print("surrogate concept:", surrogate)
    print("update(gate/mapping) concept:", updates)

    ################# surrogate buffer (mapping concept of Eq.5: V_map, mu_map) #################
    register_buffer_fn = "stacked_surrogate.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_surrogate = nice_util_SD3.get_registered_buffer(
        args, module_name_list_all, org_modules_all, st_timestep, end_timestep, n_avail_tokens,
        surrogate, embeddings_surrogate_sel_base, embeddings_surrogate_sel_base_pooled, embedding_unconditional,
        pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func)

    Vh_sur_dict = dict()
    surrogate_mean_dict = dict()
    for find_name in args.find_module_name:
        Vh_sur_dict[find_name] = dict()
        surrogate_mean_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:
        buf = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]
        n_sum = buf['n_forward'] * buf['n_sum_per_forward']

        stacked_buffer_surrogate = buf['data'] / n_sum
        stacked_buffer_surrogate_mean = buf["data_mean"] / n_sum
        stacked_buffer_surrogate_cov = stacked_buffer_surrogate - stacked_buffer_surrogate_mean.T @ stacked_buffer_surrogate_mean

        _, S_sur, Vh_sur = torch.linalg.svd(stacked_buffer_surrogate_cov, full_matrices=False)
        Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_sur
        surrogate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_surrogate_mean

        gloce_module.lora_degen.weight.data = Vh_sur[:degen_rank].T.contiguous()
        gloce_module.bias.weight.data = stacked_buffer_surrogate_mean.unsqueeze(0).clone().contiguous()

    ################# target buffer (V_tar, mu_tar) #################
    register_buffer_fn = "stacked_target.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_target = nice_util_SD3.get_registered_buffer(
        args, module_name_list_all, org_modules_all, st_timestep, end_timestep, n_avail_tokens,
        targets, embeddings_target_sel_base, embeddings_target_sel_base_pooled, embedding_unconditional,
        pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func)

    target_mean_dict = dict()
    target_cov_dict = dict()
    Vh_tar_dict = dict()
    for find_name in args.find_module_name:
        target_mean_dict[find_name] = dict()
        Vh_tar_dict[find_name] = dict()
        target_cov_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:
        buf = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]
        n_sum = buf['n_forward'] * buf['n_sum_per_forward']

        stacked_buffer_target_mean = buf['data_mean'] / n_sum
        stacked_buffer_target = buf['data'] / n_sum
        stacked_buffer_target_cov = stacked_buffer_target - stacked_buffer_target_mean.T @ stacked_buffer_target_mean

        _, S_tar, Vh_tar = torch.linalg.svd(stacked_buffer_target_cov, full_matrices=False)
        Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_tar[:update_rank]
        target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_mean
        target_cov_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_cov

    for gloce_module in network.gloce_layers:
        Vh_upd = Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name][:update_rank]
        target_mean = target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name].squeeze(0)
        dim_emb = Vh_upd.size(1)

        Vh_sur = Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name][:degen_rank]
        gloce_module.lora_update.weight.data = (Vh_sur @ (torch.eye(dim_emb).to(DEVICE_CUDA) - Vh_upd.T @ Vh_upd)).T.contiguous()
        gloce_module.debias.weight.data = target_mean.unsqueeze(0).unsqueeze(0).clone().contiguous()

    ############## v2-style gate: surrogate normalized basis (Vh_sur_norm) ##############
    # data_norm of the surrogate sum buffer holds normalized-row covariance (raw+pooled pyramid).
    Vh_sur_norm_dict = dict()
    for find_name in args.find_module_name:
        Vh_sur_norm_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:
        buf = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]
        _, S, Vh = torch.linalg.svd(buf['data_norm'], full_matrices=False)
        Vh_sur_norm_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh[:update_rank]

    ############## v2-style gate: target residual basis (discriminative) ##############
    register_buffer_fn = "stacked_target_rel.pt"
    register_func = "register_sum_buffer_avg_spatial_rel"

    buffer_sel_basis_target_rel = nice_util_SD3.get_registered_buffer(
        args, module_name_list_all, org_modules_all, st_timestep, end_timestep, n_avail_tokens,
        targets, embeddings_target_sel_base, embeddings_target_sel_base_pooled, embedding_unconditional,
        pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func,
        sur_buffer=buffer_sel_basis_surrogate, Vh_sur_dict=Vh_sur_norm_dict)

    Vh_gate_dict = dict()
    for find_name in args.find_module_name:
        Vh_gate_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:
        buf = buffer_sel_basis_target_rel[gloce_module.find_name][gloce_module.gloce_org_name]
        _, S, Vh = torch.linalg.svd(buf['data_norm'], full_matrices=False)
        Vh_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh

    ############## gate importance stats (pooled minmax v2): target & anchors ##############
    register_buffer_fn = "norm_target_rel.pt"
    register_func = "register_norm_buffer_avg_spatial_rel_stat_minmax_stack_v2"

    buffer_norm_basis_target = nice_util_SD3.get_registered_buffer(
        args, module_name_list_all, org_modules_all, st_timestep, end_timestep, n_avail_tokens,
        targets, embeddings_target_sel_base, embeddings_target_sel_base_pooled, embedding_unconditional,
        pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func,
        Vh_dict=Vh_gate_dict, Vh_sur_dict=Vh_sur_norm_dict, gate_rank=gate_rank)

    register_buffer_fn = "norm_anchor_rel.pt"
    register_func = "register_norm_buffer_avg_spatial_rel_stat_minmax_stack_v2"

    buffer_norm_basis_anchor = nice_util_SD3.get_registered_buffer(
        args, module_name_list_all, org_modules_all, st_timestep, end_timestep, n_avail_tokens,
        anchors, embeddings_anchor_sel_base, embeddings_anchor_sel_base_pooled, embedding_unconditional,
        pipe, DEVICE_CUDA, register_buffer_path, register_buffer_fn, register_func,
        Vh_dict=Vh_gate_dict, Vh_sur_dict=Vh_sur_norm_dict, gate_rank=gate_rank)

    ############## Determine gate parameters ##############
    for gloce_module in network.gloce_layers:
        buf_t = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]
        buf_a = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]

        importance_tgt = buf_t['data_max'] / buf_t['n_forward']
        importance_anc = buf_a['data_max'] / buf_a['n_forward']

        importance_tgt_stack = torch.cat([imp.unsqueeze(0) for imp in buf_t['data_stack']], dim=0)
        importance_anc_stack = torch.cat([imp.unsqueeze(0) for imp in buf_a['data_stack']], dim=0)

        print(gloce_module.gloce_org_name)
        print(f"Relative importance", (importance_tgt / importance_anc).item())

        tol1 = args.thresh
        x_center = importance_anc_stack.mean() + tol1 * importance_anc_stack.std()
        tol2 = 0.001 * max(tol1, 1.0)

        c_right = torch.tensor([0.99]).to(DEVICE_CUDA)
        C_right = torch.log(1 / (1 / c_right - 1))

        imp_center = x_center
        imp_slope = C_right / tol2

        print(f"{importance_anc_stack.max().item():10.5f}, {imp_center.item():10.5f}, {importance_tgt_stack.min().item():10.5f}, {importance_tgt_stack.max().item():10.5f}")

        Vh_gate = Vh_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name][:gate_rank]
        Vh_res = Vh_sur_norm_dict[gloce_module.find_name][gloce_module.gloce_org_name]

        gloce_module.selector.select_weight.weight.data = Vh_gate.T.unsqueeze(0).clone().contiguous()
        gloce_module.selector.select_residue.weight.data = Vh_res.T.unsqueeze(0).clone().contiguous()

        gloce_module.selector.imp_center = imp_center
        gloce_module.selector.imp_slope = imp_slope

        print()

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
    parser.add_argument("--find_module_name", type=str, default="sd3_self_out")

    parser.add_argument('--n_target_concepts', type=int, default=1)
    parser.add_argument('--n_anchor_concepts', type=int, default=3)
    parser.add_argument('--tar_concept_idx', type=int, default=0)
    parser.add_argument('--st_timestep', type=int, default=3)
    parser.add_argument('--end_timestep', type=int, default=8)
    parser.add_argument('--n_generation_per_concept', type=int, default=8)
    parser.add_argument('--param_cache_path', type=str, default="./importance_cache/org_comps/sd_v3")
    parser.add_argument('--emb_cache_path', type=str, default="./importance_cache/text_embs/sd_v3")
    parser.add_argument('--emb_cache_fn', type=str, default="text_emb_cache.pt")
    parser.add_argument("--buffer_path", type=str, default="./importance_cache/buffers")
    parser.add_argument("--use_emb_cache", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--last_layer", type=str, default="")
    parser.add_argument("--thresh", type=float, default=2.5)

    args = parser.parse_args()

    main(args)
