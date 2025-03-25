# Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation (CVPR 2025)

### ✏️ [Project Page](https://hyun1a.github.io/gloce.io) | 📄 [Paper](https://arxiv.org/abs/2503.12356)


## TODO
- [ ] Detailed explanation of GLoCE in README
- [ ] Code for Evaluation
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
