# GLoCE on Stable Diffusion 3 (DiT / MMDiT)

This document describes the SD3 implementation of GLoCE, which reproduces the localized
celebrity erasure results reported in Appendix D.3 of the paper
("Queen Elizabeth alongside Oprah Winfrey in the shot", "Barack Obama alongside Bob Marley
in the shot" — the target celebrity is replaced while the other celebrity, clothing, and
pose are preserved).

It is a port of the SD v1.4 implementation to `StableDiffusion3Pipeline`
(`stabilityai/stable-diffusion-3-medium-diffusers`, T5 disabled). All SD3-specific code
carries the `_sd3` suffix; the SD v1.4 code in this repository is unchanged and both
pipelines coexist in the same codebase.

## Setup

The SD3 pipeline requires newer packages than the SD v1.4 pipeline (in particular
`diffusers>=0.31` for `StableDiffusion3Pipeline`), so use a separate environment:

```bash
conda create -n GLoCE_sd3 python=3.10
conda activate GLoCE_sd3
pip install -r requirements_sd3.txt
```

Access to `stabilityai/stable-diffusion-3-medium-diffusers` on Hugging Face is gated;
log in with `huggingface-cli login` first.

## Running

```bash
# 1) update (training-free parameter estimation)
#    target concept index: 1 = barack obama, 301 = queen elizabeth
#    (see configs/train_celeb/prompt_train_gloce_sd3_target.yaml)
bash shell_scripts/celebs/update_gloce_sd3.sh 1
bash shell_scripts/celebs/update_gloce_sd3.sh 301

# 2) generation (erased model vs base SD3)
bash shell_scripts/celebs/generate_by_gloce_sd3.sh
```

Or directly:

```bash
python generate/generate_by_gloce_sd3.py \
  --model_paths output/celeb_sd3/sd3_block_context/ur16_dr2_gr4_st3_end8_gen8_th2.5/barack_obama \
  --find_module_name sd3_block_context --gate_rank 4 --update_rank 16 --degen_rank 2 \
  --last_layer "transformer_blocks.22" --st_step 3 --n_step 28 \
  --prompts "Barack Obama alongside Bob Marley in the shot" \
  --seeds 1,2,3,4,5,6,7,8 --steps 28 --guidance 7.0 --size 512 \
  --out_dir generated_images/obama_erased

# base SD3 for comparison: same command without --model_paths
```

## How GLoCE is applied to the MMDiT architecture

**1. Which DiT / MMDiT layers are selected for GLoCE module insertion?**
GLoCE modules are attached to every `JointTransformerBlock` of `SD3Transformer2DModel`
whose `context_pre_only == False` (blocks 0–22 of the 24-block MMDiT in SD3-medium).
See `find_module_name = "sd3_block_context"` in `src/engine/nice_util_sd3.py::get_modules_list`.

**2. Is the GLoCE gate applied to image tokens, text tokens, or joint hidden states?**
To the **text-token half of the joint residual stream**: GLoCE wraps each block's forward
and modifies the returned `encoder_hidden_states` (text tokens) with
`x' = (1-s(x))·x + s(x)·η(µ_map + V̂map V̂mapᵀ(I − V̂tar V̂tarᵀ)(x − µ_tar))` (paper Eq. 5–8).
The gate `s(x)` is evaluated per text token. Modifying the text residual stream is essential
for the DiT architecture: the target-concept information enters the image tokens only through
joint attention from the text stream, and once erased from the text stream it cannot re-enter.
(Image-side variants — `attn.to_out.0` outputs or the image hidden states — leave the identity
information in the residual stream and do NOT erase the concept; we verified this empirically.)

**3. How does the SD-UNet module selection map to the SD3 transformer?**
In SD v1.4 GLoCE is inserted at `attn2.to_out.0` (cross-attention output), i.e. at the point
where text conditioning is injected into the image features. In SD3 there is no cross-attention;
the equivalent injection point is the text stream of the joint transformer block, so GLoCE is
applied to the per-block `encoder_hidden_states` output.

**4. Configuration used for the SD3 experiments**
- Model: `stabilityai/stable-diffusion-3-medium-diffusers`, `text_encoder_3=None` (no T5), fp32
- Update ranks: r1 (mapping/degen) = 2, r2 (target/update) = 16, gate rank r3 = 4
- Surrogate/mapping concept: "a celebrity"; gate-basis concept ("a person") for stats
- Anchors: 3 most similar celebrities from the concept pool (auto-selected by CLIP text similarity)
- Statistics collection: 8 generations per concept, transformer forwards 3–8 of a 28-step
  flow-matching schedule (the analog of DDIM steps 10–20/50 in the SD1.4 setup)
- Gate calibration: importance p(X) computed per text token (tokens 1..8) as
  ‖V̂gateᵀ · r(x)‖² where r(x) is the normalized residual after removing the surrogate basis;
  sigmoid center = mean + 2.5·std of anchor importance, slope from u = 0.99
- Inference: η = 1.0, gate active from step 3 (st_step=3, n_step=28), 28 steps, CFG 7.0, 512×512

## SD3-specific files (`_sd3` suffix)

- `update/update_gloce_sd3.py` — SD3 port of `update/update_gloce.py`
  (training-free: collects activation statistics with forward hooks during partial generations,
  then sets all GLoCE parameters in closed form).
- `generate/generate_by_gloce_sd3.py` — generation with the erased / base SD3 model.
- `src/models/gloce_sd3.py` — GLoCE modules for SD3: `forward_sd3_block` wrapper for
  `JointTransformerBlock` (gates `encoder_hidden_states`) and a per-token gate
  (normalize → remove surrogate-basis projection → renormalize → squared projection onto
  the discriminative basis → sigmoid). The SD v1.4 modules in `src/models/gloce.py`
  are untouched.
- `src/engine/gloce_register_buffer_sd3.py` — register hooks for SD3: for
  `sd3_block_context` the hook reads `output[0]` (text half of the joint block output)
  and collects statistics over text tokens `[1:1+n_tokens]`.
- `src/engine/nice_util_sd3.py`, `src/engine/train_util_sd3.py` — SD3 utilities
  (module discovery, partial-generation loops, text-embedding cache).
- `configs/train_celeb/config_train_gloce_sd3.yaml`,
  `configs/train_celeb/prompt_train_gloce_sd3_{target,anchor,update}.yaml` — SD3 configs.
  SD3 does not generate all celebrities reliably, so the target/anchor pools were
  pre-filtered accordingly ("queen elizabeth" is target index 301).
- `shell_scripts/celebs/update_gloce_sd3.sh`, `shell_scripts/celebs/generate_by_gloce_sd3.sh`
- `requirements_sd3.txt`
