# Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation (CVPR 2025)

### ✏️ [Project Page](https://hyun1a.github.io/gloce.io) | 📄 [Paper](https://arxiv.org/abs/2503.12356)


## TODO
- [ ] Detailed explanation of GLoCE in README
- [x] Code for Evaluation
- [ ] Generation with Stable Diffusion v3



## Setup for experiments

**OS**: Ubuntu 20.04.5 LTS

**Python**: 3.10.14

<pre>
conda create -n GLoCE python=3.10
</pre>

Please install packages in requirements.txt
<pre>
pip install -r requirements.txt
</pre>

## Running Experiments
### Celebrities Erasure
**Update:**
<pre>
bash ./shell_scripts/celebs/update_gloce.sh
</pre>

**Generation:**
<pre>
bash ./shell_scripts/celebs/generate_by_gloce.sh
</pre>

- For generation of diverse domains such as artistic styles or characters from celebrity-erased model, please change the variable GEN_CONFIG in
   ./shell_scripts/celebs/generate_by_gloce.sh (config files for different domains are listed)



## Explicit Contents Erasure
**Update:**
<pre>
sh ./shell_scripts/explicit/update_gloce.sh
</pre>

**Generation**
<pre>
sh ./shell_scripts/explicit/generate_by_gloce.sh
</pre>

- For generation of diverse domains such as celebrites or characters from explicit contents-erased model, 
   please change the variable GEN_CONFIG in ./shell_scripts/artists/generate_by_gloce.sh


### Artistic Styles Erasure
**Update:**
<pre>
bash ./shell_scripts/artists/update_gloce.sh
</pre>

**Generation:**
<pre>
bash ./shell_scripts/artists/generate_by_gloce.sh
</pre>

- For generation of diverse domains such as celebrites or characters from styles-erased model, 
   please change the variable GEN_CONFIG in ./shell_scripts/artists/generate_by_gloce.sh (config files for different domains are listed)



## Evaluation Metrics


In our paper, GLoCE, we utilize various metrics including [FID](https://github.com/GaParmar/clean-fid)(Fréchet Inception Distance), [KID](https://github.com/GaParmar/clean-fid)(Kernel Inception Distance), [CLIP score](https://github.com/openai/CLIP), [GIPHY Celebrity Detector](https://github.com/Giphy/celeb-detection-oss), and [NudeNet Detector](https://pypi.org/project/nudenet/) for explicit images.

**Evaluate FID / KID**
<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_fid_kid.sh
</pre>

**Evaluate CLIP Score**
<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_clip_score.sh
</pre>

**Evaluate GIPHY Celebrity Detector**

To use the GIPHY Celebrity Detector, download the official GCD code and create a conda environment for GCD by following the [official guide](https://github.com/Giphy/celeb-detection-oss). 
(Note that the GCD Python environment is not compatible with the GLoCE environment.) 
After setting up the GCD environment, please refer to [our installation guide](https://github.com/Hyun1A/GLoCE/tree/main/metrics) in the 'metrics' folder.

<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_giphy_score.sh
</pre>

**Evaluate NudeNet Detector**
<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_I2P.sh
</pre>


## Acknowledgements
We thank the following contributors that our code is based on: [SPM](https://github.com/Con6924/SPM?tab=readme-ov-file), [MACE](https://github.com/Shilin-LU/MACE?tab=readme-ov-file).

## Citation
If you find the repo useful, please consider citing.

<pre>
@InProceedings{lee2025localized,
    author    = {Lee, Byung Hyun and Lim, Sungjin and Chun, Se Young},
    title     = {Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation},
    booktitle = {CVPR},
    year      = {2025},
}
</pre>