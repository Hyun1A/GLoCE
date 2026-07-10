# SD3 generation with official-port GLoCE network (see update/update_gloce_sd3.py)
import argparse
import os, sys, random
from pathlib import Path

import numpy as np
import torch

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

from src.configs.config import parse_precision
from src.models.merge_gloce import load_state_dict
from src.models.gloce_sd3 import (
    GLoCELayerOutProp,
    GLoCENetworkOutProp,
)
import src.engine.nice_util_sd3 as nice_util_SD3
from diffusers import StableDiffusion3Pipeline

device = torch.device('cuda:0')


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(args):
    weight_dtype = parse_precision(args.precision)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32,
        text_encoder_3=None, tokenizer_3=None).to(device)

    find_module_names = args.find_module_name.split(",")

    org_modules_all = []
    module_name_list_all = []
    for find_module_name in find_module_names:
        module_name, module_type = nice_util_SD3.get_module_name_type(find_module_name)
        org_modules, module_name_list = nice_util_SD3.get_modules_list(
            pipe.transformer, pipe.text_encoder, find_module_name, module_name, module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)

    network = None
    if args.model_paths:
        model_paths = [Path(os.path.join(mp, "ckpt.safetensors")) for mp in args.model_paths]
        cpes, metadatas = zip(*[load_state_dict(mp, weight_dtype) for mp in model_paths])

        network = GLoCENetworkOutProp(
            pipe.transformer,
            pipe.text_encoder,
            multiplier=1.0,
            alpha=float(metadatas[0]["alpha"]),
            module=GLoCELayerOutProp,
            degen_rank=args.degen_rank,
            gate_rank=args.gate_rank,
            update_rank=args.update_rank,
            n_concepts=len(model_paths),
            org_modules_all=org_modules_all,
            module_name_list_all=module_name_list_all,
            find_module_names=find_module_names,
            last_layer=args.last_layer,
            st_step=args.st_step,
            n_step=args.n_step,
        ).to(device, dtype=weight_dtype)

        for n_concept in range(len(cpes)):
            for idx, (k, m) in enumerate(network.named_modules()):
                if m.__class__.__name__ == "GLoCELayerOutProp":
                    m.eta = args.eta
                    if args.no_gate:
                        m.use_gate = False
                    for k_child, m_child in m.named_children():
                        module_name = f"{k}.{k_child}"
                        if ("lora_update" in k_child) or ("lora_degen" in k_child):
                            m_child.weight.data[n_concept] = cpes[n_concept][module_name + '.weight']
                        elif "bias" in k_child:
                            m_child.weight.data[:, n_concept:n_concept + 1, :] = cpes[n_concept][module_name + '.weight']
                        elif "selector" in k_child:
                            m_child.select_weight.weight.data[n_concept] = cpes[n_concept][module_name + '.select_weight.weight'].squeeze(0)
                            m_child.select_mean_diff.weight.data[n_concept] = cpes[n_concept][module_name + '.select_mean_diff.weight'].squeeze(0)
                            m_child.select_residue.weight.data[n_concept] = cpes[n_concept][module_name + '.select_residue.weight'].squeeze(0)
                            m_child.imp_center[n_concept] = args.center_scale * cpes[n_concept][module_name + '.imp_center']
                            m_child.imp_slope[n_concept] = cpes[n_concept][module_name + '.imp_slope']

        network.to(device, dtype=weight_dtype)
        network.eval()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    negative_prompt = args.negative_prompt if args.negative_prompt else None

    for prompt in args.prompts:
        for seed in seeds:
            tag = "gloce" if network is not None else "base"
            fn = f"{args.out_dir}/{tag}_seed{seed}_{prompt.replace(' ', '_')}.png"
            if os.path.isfile(fn):
                print("skip", fn)
                continue
            seed_everything(seed)
            gen_kwargs = dict(
                prompt=[prompt],
                negative_prompt=negative_prompt,
                num_inference_steps=args.steps,
                height=args.size,
                width=args.size,
                guidance_scale=args.guidance,
                num_images_per_prompt=1,
                generator=torch.cuda.manual_seed(seed),
            )
            if network is not None:
                # reset per-layer step counters for reproducible gating
                for gl in network.gloce_layers:
                    gl.t_counter = 0
                with torch.no_grad():
                    with network:
                        images = pipe(**gen_kwargs).images
            else:
                with torch.no_grad():
                    images = pipe(**gen_kwargs).images
            images[0].save(fn)
            print("saved", fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="*", default=None, help="Dirs containing ckpt.safetensors; omit for base SD3")
    parser.add_argument("--find_module_name", type=str, default="sd3_self_out")
    parser.add_argument("--gate_rank", type=int, default=1)
    parser.add_argument("--update_rank", type=int, default=16)
    parser.add_argument("--degen_rank", type=int, default=2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--last_layer", type=str, default="transformer_blocks.23.attn.to_out.0")
    parser.add_argument("--st_step", type=int, default=3)
    parser.add_argument("--n_step", type=int, default=28)
    parser.add_argument("--center_scale", type=float, default=1.0)
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--out_dir", type=str, default="./repro_images")
    args = parser.parse_args()
    main(args)
