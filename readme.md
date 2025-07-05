# Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation (CVPR 2025)

### ✏️ [Project Page](https://hyun1a.github.io/gloce.io) | 📄 [Paper](https://arxiv.org/abs/2503.12356)

> **Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation**<br>
> Byung Hyun Lee, Sungjin Lim, Se Young Chun <br>
> 
>**Abstract**: Fine-tuning based concept erasing has demonstrated promising results in preventing generation of harmful contents from text-to-image diffusion models by removing target concepts while preserving remaining concepts. To maintain the generation capability of diffusion models after concept erasure, it is necessary to remove only the image region containing the target concept when it locally appears in an image, leaving other regions intact. However, prior arts often compromise fidelity of the other image regions in order to erase the localized target concept appearing in a specific area, thereby reducing the overall performance of image generation. To address these limitations, we first introduce a framework called localized concept erasure, which allows for the deletion of only the specific area containing the target concept in the image while preserving the other regions. As a solution for the localized concept erasure, we propose a training-free approach, dubbed Gated Low-rank adaptation for Concept Erasure (GLoCE), that injects a lightweight module into the diffusion model. GLoCE consists of low-rank matrices and a simple gate, determined only by several generation steps for concepts without training. By directly applying GLoCE to image embeddings and designing the gate to activate only for target concepts, GLoCE can selectively remove only the region of the target concepts, even when target and remaining concepts coexist within an image. Extensive experiments demonstrated GLoCE not only improves the image fidelity to text prompts after erasing the localized target concepts, but also outperforms prior arts in efficacy, specificity, and robustness by large margin and can be extended to mass concept erasure.
<br>


![overview](./assets/figure_intro.png)
Illustration of overall results of concept erasing after erasing 50 celebrities by a baseline (MACE) and ours. To preserve generation capability after concept erasing, it is essential to maintain high fidelity for remaining concepts even when target concepts are included in same text prompts. However, baselines often struggle to achieve the fidelity. The proposed method, GLoCE, significantly improves this fidelity while demonstrating strong performance in efficacy, specificity, and robustness, which are key conditions for effective erasure.
<br>
<br>



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


## Artistic Styles Erasure
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

To use the GIPHY Celebrity Detector, download the official GCD code and create a conda environment for giphy by following [our installation guide](https://github.com/Hyun1A/GLoCE/tree/main/metrics) in the 'metrics' folder.

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
