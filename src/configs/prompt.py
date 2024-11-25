from typing import Literal, Optional, Union

import yaml
from pathlib import Path
import pandas as pd
import random
import numpy as np

from pydantic import BaseModel, root_validator
from transformers import CLIPTextModel, CLIPTokenizer
import torch

from src.misc.clip_templates import imagenet_templates
from src.engine.train_util import encode_prompts

from .prompt_util import smooth_tensor

ACTION_TYPES = Literal[
    "erase",
    "erase_with_la",
]

class PromptEmbedsXL:
    text_embeds: torch.FloatTensor
    pooled_embeds: torch.FloatTensor

    def __init__(self, embeds) -> None:
        self.text_embeds, self.pooled_embeds = embeds

PROMPT_EMBEDDING = Union[torch.FloatTensor, PromptEmbedsXL]


class PromptEmbedsCache:
    
    prompts = {}

    def __setitem__(self, __name, __value):
        self.prompts[__name] = __value

    def __getitem__(self, __name: str):
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class PromptSettings(BaseModel):  # yaml
    target: Union[str, list]
    positive: Union[str, list] = None  # if None, target will be used
    unconditional: str = ""  # default is ""
    neutral: str = None  # if None, unconditional will be used
    action: ACTION_TYPES = "erase"  # default is "erase"
    guidance_scale: float = 1.0  # default is 1.0
    resolution: int = 512  # default is 512
    dynamic_resolution: bool = False  # default is False
    batch_size: int = 1  # default is 1
    dynamic_crops: bool = False  # default is False. only used when model is XL
    use_template: bool = False  # default is False
    
    la_strength: float = 1000.0
    sampling_batch_size: int = 4

    seed: int = None
    case_number: int = 0

    @root_validator(pre=True)
    def fill_prompts(cls, values):
        keys = values.keys()
        if "target" not in keys:
            raise ValueError("target must be specified")
        if "positive" not in keys:
            values["positive"] = values["target"]
        if "unconditional" not in keys:
            values["unconditional"] = ""
        if "neutral" not in keys:
            values["neutral"] = values["unconditional"]

        return values


class PromptEmbedsPair:
    target: PROMPT_EMBEDDING  # the concept that do not want to generate 
    positive: PROMPT_EMBEDDING  # generate the concept
    unconditional: PROMPT_EMBEDDING  # uncondition (default should be empty)
    neutral: PROMPT_EMBEDDING  # base condition (default should be empty)
    use_template: bool = False  # use clip template or not

    guidance_scale: float
    resolution: int
    dynamic_resolution: bool
    batch_size: int
    dynamic_crops: bool

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: PROMPT_EMBEDDING,
        positive: PROMPT_EMBEDDING,
        unconditional: PROMPT_EMBEDDING,
        neutral: PROMPT_EMBEDDING,
        settings: PromptSettings,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral
        
        self.settings = settings

        self.use_template = settings.use_template
        self.guidance_scale = settings.guidance_scale
        self.resolution = settings.resolution
        self.dynamic_resolution = settings.dynamic_resolution
        self.batch_size = settings.batch_size
        self.dynamic_crops = settings.dynamic_crops
        self.action = settings.action
        
        self.la_strength = settings.la_strength
        self.sampling_batch_size = settings.sampling_batch_size
        
        
    def _prepare_embeddings(
        self, 
        cache: PromptEmbedsCache,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
    ):
        """
        Prepare embeddings for training. When use_template is True, the embeddings will be
        format using a template, and then be processed by the model.
        """
        if not self.use_template:
            return
        template = random.choice(imagenet_templates)
        target_prompt = template.format(self.settings.target)
        if cache[target_prompt]:
            self.target = cache[target_prompt]
        else:
            self.target = encode_prompts(tokenizer, text_encoder, [target_prompt])
        
    
    def _erase(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        **kwargs,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""

        erase_loss = self.loss_fn(
            target_latents,
            neutral_latents
            - self.guidance_scale * (positive_latents - neutral_latents),
        )
        
        erase_naive = ( (target_latents-neutral_latents)**2 ).detach()

        
        losses = {
            "loss": erase_loss,
            "loss/erase": erase_loss,
            "loss/erase_naive": erase_naive.mean(),
        }
        return losses
        
        
    def _erase_with_la(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        anchor_latents: torch.FloatTensor, 
        anchor_latents_ori: torch.FloatTensor, 
        **kwargs,
    ):
        anchoring_loss = self.loss_fn(anchor_latents, anchor_latents_ori)
        anchoring_naive = ( (anchor_latents-anchor_latents_ori)**2 ).detach().mean()
        
        
        erase_losses = self._erase(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
        )
        
        
        losses = {
            "loss": erase_losses["loss/erase"] + self.la_strength * anchoring_loss,
            "loss/erase": erase_losses["loss/erase"],
            "loss/anchoring": anchoring_loss,
            "loss/erase_naive": erase_losses["loss/erase_naive"],
            "loss/anchoring_naive": anchoring_naive
        }        

        
        
        # losses = {
        #     "loss": erase_loss + self.la_strength * anchoring_loss,
        #     "loss/erase": erase_loss,
        #     "loss/anchoring": anchoring_loss
        # }
        
        
        return losses

    def loss(
        self,
        **kwargs,
    ):
        if self.action == "erase":
            return self._erase(**kwargs)
        elif self.action == "erase_with_la":
            return self._erase_with_la(**kwargs)
        else:
            raise ValueError("action must be erase or erase_with_la")
            
            
            
            
    def _erase_pert(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        target_latents_pert: torch.FloatTensor,
        neutral_latents: torch.FloatTensor,  # ""
        delta: float,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""
        
        noisy_latent_cov = ( (target_latents.detach() - target_latents_pert.detach()) / delta )**2
        noisy_latent_cov = torch.clip(noisy_latent_cov, 0.01, 100)
                
        erase_loss = (target_latents-neutral_latents)**2 / noisy_latent_cov

        erase_naive = ( (target_latents-neutral_latents)**2 ).detach()
        

        losses = {
            "loss": erase_loss.mean(),
            "loss/erase": erase_loss.mean(),
            "loss/erase_naive": erase_naive.mean()
        }
        return losses            
            
        
            
    def _anchor_pert(
        self,
        anchor_latents: torch.FloatTensor,  # "van gogh"
        anchor_latents_ori: torch.FloatTensor,  # "van gogh"
        anchor_latents_pert: torch.FloatTensor,  # ""
        delta: float,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""


        noisy_latent_cov = ( (anchor_latents_pert.detach()-anchor_latents.detach()) / delta )**2
        noisy_latent_cov = torch.clip(noisy_latent_cov, 0.01, 100)
        
        loss = (anchor_latents-anchor_latents_ori)**2 / noisy_latent_cov

        anchoring_naive = ( (anchor_latents-anchor_latents_ori)**2 ).detach()
        
        losses = {
            "loss/anchoring": loss.mean(),
            "loss/anchoring_naive": anchoring_naive.mean()
        }        
        
        return losses
            
    def _erase_with_la_pert(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        target_latents_pert: torch.FloatTensor,
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        anchor_latents: torch.FloatTensor, 
        anchor_latents_ori: torch.FloatTensor, 
        anchor_latents_pert: torch.FloatTensor, 
        delta: float,
    ):
        
        
        anchoring_losses = self._anchor_pert(anchor_latents,
                                             anchor_latents_ori, 
                                             anchor_latents_pert, 
                                             delta)
        
        erase_losses = self._erase_pert(
            target_latents=target_latents,
            positive_latents=positive_latents,
            target_latents_pert=target_latents_pert,
            neutral_latents=neutral_latents,
            delta=delta,
        )
                
        
        losses = {
            "loss": erase_losses["loss/erase"] + self.la_strength * anchoring_losses["loss/anchoring"],
            "loss/erase": erase_losses["loss/erase"],
            "loss/anchoring": anchoring_losses["loss/anchoring"],
            "loss/erase_naive": erase_losses["loss/erase_naive"],
            "loss/anchoring_naive": anchoring_losses["loss/anchoring_naive"]
        }
        return losses            
            
            
    def loss_pert(
        self,
        **kwargs,
    ):
        if self.action == "erase_with_la":
            return self._erase_with_la_pert(**kwargs)

    
            
    def _erase_pert_inverse(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        target_latents_pert: torch.FloatTensor,
        neutral_latents: torch.FloatTensor,  # ""
        delta: float,
        smoothing: bool,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""
        
        noisy_latent_cov = ( (target_latents.detach() - target_latents_pert.detach()) / delta )**2
        noisy_latent_cov = torch.clip(noisy_latent_cov, 0.01, 100)

        if smoothing:
            noisy_latent_cov = smooth_tensor (torch.clip(noisy_latent_cov, 0.01, 100) )
        
        
        surrogate_latents = neutral_latents- self.guidance_scale * (positive_latents - neutral_latents)
        
        # erase_loss = heat_map_erase * (target_latents-surrogate_latents)**2
        
        erase_loss = noisy_latent_cov * (target_latents-surrogate_latents)**2

        erase_naive = ( (target_latents-neutral_latents)**2 ).detach()
        
        self.noisy_latent_cov_erase = noisy_latent_cov
        
        np.save("erase.npy", noisy_latent_cov.numpy())
        
        
        losses = {
            "loss": erase_loss.mean(),
            "loss/erase": erase_loss.mean(),
         
            "loss/erase_naive": erase_naive.mean()
        }
        
        return losses            
            
        
            
    def _anchor_pert_inverse(
        self,
        anchor_latents: torch.FloatTensor,  # "van gogh"
        anchor_latents_ori: torch.FloatTensor,  # "van gogh"
        anchor_latents_pert: torch.FloatTensor,  # ""
        delta: float,
        smoothing: bool,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""


        noisy_latent_cov = ( (anchor_latents_pert.detach()-anchor_latents.detach()) / delta )**2
        noisy_latent_cov = torch.clip(noisy_latent_cov, 0.01, 100)

        if smoothing:
            noisy_latent_cov = smooth_tensor (torch.clip(noisy_latent_cov, 0.01, 100) )
        
        loss = noisy_latent_cov * (anchor_latents-anchor_latents_ori)**2

        anchoring_naive = ( (anchor_latents-anchor_latents_ori)**2 ).detach()
        
        self.noisy_latent_cov_anchor = noisy_latent_cov
        
        np.save("anchor.npy", noisy_latent_cov.numpy())
        
        losses = {
            "loss/anchoring": loss.mean(),
            "loss/anchoring_naive": anchoring_naive.mean()
        }        
        
        return losses
            
    def _erase_with_la_pert_inverse(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        target_latents_pert: torch.FloatTensor,
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        anchor_latents: torch.FloatTensor, 
        anchor_latents_ori: torch.FloatTensor, 
        anchor_latents_pert: torch.FloatTensor, 
        delta: float,
        smoothing=False,
    ):
        
        
        anchoring_losses = self._anchor_pert_inverse(anchor_latents,
                                             anchor_latents_ori, 
                                             anchor_latents_pert, 
                                             delta,
                                             smoothing,)
        
        erase_losses = self._erase_pert_inverse(
            target_latents=target_latents,
            positive_latents=positive_latents,
            target_latents_pert=target_latents_pert,
            neutral_latents=neutral_latents,
            delta=delta,
            smoothing=smoothing,
        )
                        
        losses = {
            "loss": erase_losses["loss/erase"] + self.la_strength * anchoring_losses["loss/anchoring"],
            "loss/erase": erase_losses["loss/erase"],
            "loss/anchoring": anchoring_losses["loss/anchoring"],
            "loss/erase_naive": erase_losses["loss/erase_naive"],
            "loss/anchoring_naive": anchoring_losses["loss/anchoring_naive"]
        }
        return losses            
            
            
    def loss_pert_inverse(
        self,
        **kwargs,
    ):
        if self.action == "erase_with_la":
            return self._erase_with_la_pert_inverse(**kwargs)
        
        
        
            
    def _erase_attn(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        heat_map_erase: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""
        
        surrogate_latents = neutral_latents- self.guidance_scale * (positive_latents - neutral_latents)
        
        erase_loss = heat_map_erase * (target_latents-surrogate_latents)**2

        # erase_loss = (target_latents-surrogate_latents)**2
        
        erase_naive = ( (target_latents-neutral_latents)**2 ).detach()

        losses = {
            "loss": erase_loss.mean(),
            "loss/erase": erase_loss.mean(),
            "loss/erase_naive": erase_naive.mean()
        }
        return losses            

    def _anchor_attn(
        self,
        anchor_latents: torch.FloatTensor,  # "van gogh"
        anchor_latents_ori: torch.FloatTensor,  # "van gogh"
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""

        loss = (anchor_latents-anchor_latents_ori)**2
        anchoring_naive = ( (anchor_latents-anchor_latents_ori)**2 ).detach()
        
        losses = {
            "loss/anchoring": loss.mean(),
            "loss/anchoring_naive": anchoring_naive.mean()
        }        
        
        return losses    
    

    
    def _erase_with_la_attn(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        anchor_latents: torch.FloatTensor, 
        anchor_latents_ori: torch.FloatTensor, 
        heat_map_erase= torch.FloatTensor,
    ):
        
        anchoring_losses = self._anchor_attn(anchor_latents,
                                     anchor_latents_ori)

        
        erase_losses = self._erase_attn(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            heat_map_erase=heat_map_erase,
        )
        
        losses = {
            "loss": erase_losses["loss/erase"] + self.la_strength * anchoring_losses["loss/anchoring"],
            "loss/erase": erase_losses["loss/erase"],
            "loss/anchoring": anchoring_losses["loss/anchoring"],
            "loss/erase_naive": erase_losses["loss/erase_naive"],
            "loss/anchoring_naive": anchoring_losses["loss/anchoring_naive"]
        }
        return losses            
            
    
    def loss_attn(
        self,
        **kwargs,
    ):
        if self.action == "erase_with_la":
            return self._erase_with_la_attn(**kwargs)
            
                
                

    

    def _push(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        anchor_latents: torch.FloatTensor, 
    ):
        
        loss = -(target_latents - anchor_latents)**2
    
        losses = {
            "loss/push": loss.mean(),
        }        
        
        return losses    
    
    
    
    
    
    def _erase_with_la_attn_push(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        anchor_latents: torch.FloatTensor, 
        anchor_latents_ori: torch.FloatTensor, 
        heat_map_erase: torch.FloatTensor,
        push_strength=1.
    ):
        
        anchoring_losses = self._anchor_attn(anchor_latents,
                                     anchor_latents_ori)

        
        erase_losses = self._erase_attn(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            heat_map_erase=heat_map_erase,
        )
        
        push_losses = self._push(
            target_latents = target_latents,  # "van gogh"
            anchor_latents = anchor_latents, 
        )
        
        
        
        losses = {
            "loss": erase_losses["loss/erase"] + self.la_strength * anchoring_losses["loss/anchoring"] + push_strength * push_losses,
            "loss/erase": erase_losses["loss/erase"],
            "loss/anchoring": anchoring_losses["loss/anchoring"],
            "loss/erase_naive": erase_losses["loss/erase_naive"],
            "loss/anchoring_naive": anchoring_losses["loss/anchoring_naive"],
            "loss/push": push_losses["loss/push"]
        }
        return losses            
            
    
    def loss_attn_push(
        self,
        **kwargs,
    ):
        if self.action == "erase_with_la":
            return self._erase_with_la_attn_push(**kwargs)
            
                
                
                
                
                
                
                
                
                                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
    


def load_prompts_from_yaml(path) -> list[PromptSettings]:
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)

    if len(prompts) == 0:
        raise ValueError("prompts file is empty")

    prompt_settings = [PromptSettings(**prompt) for prompt in prompts]

    return prompt_settings

def load_prompts_from_table(path) -> list[PromptSettings]:
    # check if the file ends with .csv
    if not path.endswith(".csv"):
        raise ValueError("prompts file must be a csv file")
    df = pd.read_csv(path)
    prompt_settings = []
    for _, row in df.iterrows():
        prompt_settings.append(PromptSettings(**dict(
            target=str(row.prompt),
            seed=int(row.get('sd_seed', row.evaluation_seed)),
            case_number=int(row.get('case_number', -1)),
        )))
    return prompt_settings

def compute_rotation_matrix(target: torch.FloatTensor):
    """Compute the matrix that rotate unit vector to target.
    
    Args:
        target (torch.FloatTensor): target vector.
    """
    normed_target = target.view(-1) / torch.norm(target.view(-1), p=2)
    n = normed_target.shape[0]
    basis = torch.eye(n).to(target.device)
    basis[0] = normed_target
    for i in range(1, n):
        w = basis[i]
        for j in range(i):
            w = w - torch.dot(basis[i], basis[j]) * basis[j]
        basis[i] = w / torch.norm(w, p=2)
    return torch.linalg.inv(basis)