from typing import Literal, Optional, Union

import yaml

from pydantic import BaseModel
import torch

PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]


class PretrainedModelConfig(BaseModel):
    name_or_path: str
    safetensor: Union[list[str], str] = None
    v2: bool = False
    v_pred: bool = False
    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    rank: int = 1
    continual_rank: int = 4
    alpha: float = 1.0
    delta: float = 1e-5
    num_embeddings: int = 3
    hidden_size: int = 128
    init_size: int = 16


class TrainConfig(BaseModel):    
    precision: PRECISION_TYPES = "float32"
    noise_scheduler: Literal["ddim", "ddpm", "lms", "euler_a"] = "ddim"

    iterations: int = 3000
    batch_size: int = 1

    lr: float = 1e-4
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5

    optimizer_type: str = "AdamW8bit"
    optimizer_args: list[str] = None

    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 500
    lr_scheduler_num_cycles: int = 3
    lr_scheduler_power: float = 1.0
    lr_scheduler_args: str = ""

    max_grad_norm: float = 0.0

    max_denoising_steps: int = 30
    importance_path: str="./"
    portion: float=1.0
    push_strength: float=1.0
    norm_strength: float=1.0
    
    pal: float=0.01
    value_weight: float=0.1
    swap_iteration: int = 1500
    erase_scale: float = 1.
    
    #########################################
    ########### For adv memory ##############
    num_stages: int = 10
    iterations_adv: int = 1000
    lr_scheduler_adv: str = "cosine_with_restarts"
    lr_warmup_steps_adv: int = 500
    lr_scheduler_num_cycles_adv: int = 3
    lr_scheduler_power_adv: float = 1.0
    lr_scheduler_args_adv: str = ""    
    lr_adv: float = 1e-4
    adv_coef: float = 0.1
    num_add_prompts: int = 10
    num_multi_concepts: int = 1
    train_seed: int = 0
    factor_init_iter: int = 1
    factor_init_lr: float = 1
    factor_init_lr_cycle: int = 1
    do_adv_learn: bool = False
    ########### For adv memory ##############
    #########################################
    
    st_prompt_idx: int = 0
    end_prompt_idx: int = 100000000
    resume_stage: int = 0
    skip_learned: bool = False
    noise_scale: float = 0.001
    mixup: bool = True
    
class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 500
    precision: PRECISION_TYPES = "float32"
    stage_interval: int = 1

class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project_name: str = "proposed_method"
    run_name: str = None
    verbose: bool = False
    
    interval: int = 50
    prompts: list[str] = []
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    # negative_prompt: str = ""    
    anchor_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = None
    generate_num: int = 1
    eval_num: int = 10
    stage_interval: int = 1
    gen_init_img: bool = False
    
class InferenceConfig(BaseModel):
    use_wandb: bool = False
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seeds: list[int] = None    
    precision: PRECISION_TYPES = "float32"

class OtherConfig(BaseModel):
    use_xformers: bool = False


class RootConfig(BaseModel):
    prompts_file: Optional[str] = None
    scripts_file: Optional[str] = None
    replace_word: Optional[str] = None
    
    prompts_file_target: Optional[str] = None   
    prompts_file_anchor: Optional[str] = None   
    prompts_file_update: Optional[str] = None
     
    pretrained_model: PretrainedModelConfig

    network: Optional[NetworkConfig] = None

    train: Optional[TrainConfig] = None

    save: Optional[SaveConfig] = None

    logging: Optional[LoggingConfig] = None

    inference: Optional[InferenceConfig] = None

    other: Optional[OtherConfig] = None


def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")


def load_config_from_yaml(config_path: str) -> RootConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.inference is None:
        root.inference = InferenceConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root
