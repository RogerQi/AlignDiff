# AlignDiff

Official implementation for ECCV2024 paper "AlignDiff: Aligning Diffusion Models for General Few-Shot Segmentation".

[[Paper](https://motion.cs.illinois.edu/papers/ECCV2024-Qiu-AlignDiff.pdf)]

## Setup dependency

Clone the codebase.

```bash
cd ~
git clone --recursive https://github.com/RogerQi/aligndiff
```

Follow [INSTALL.md](./docs/INSTALL.md) to install required dependencies, and [PREPARE.md](.docs/PREPARE.md) to set up
huggingface accounts and download sample FSS weights.

## Step 1: build a prompt bank with texts or masked inversion

As decribed in the paper, for the same category, we introduce a prompt bank to make
generated samples more diverse. There are three planned sources of prompts:

1. Texts (manually entered simple nouns or with detailed descriptions)
2. (TODO) Augmented texts (retrieval from a database or augmented using LLMs)
3. Learned per-object embedding (proposed normalized textual inversion)

AlignDiff requires the prompts to be organized in a directory to be used. An example
layout containing two categories and a JSON file is provided in `./examples_dataset/`.

To optimize the per-object embedding for image-mask pairs in the provided datasets, run

```bash
python -m scripts.image_mask_inversion --dataset_dir ./example_dataset
```

On a single RTX4090, this step takes ~3mins.

**Tips:** the invesion process is quite sensitive to `bg_lambda`. If it ends up generating degenerate samples, you can try tweaking this parameter.

## Step 2: generate images and (coarse) masks

We use the following command to generate images and masks.

```bash
python -m scripts.diffuse_img_attention --dataset_dir ./example_dataset --image_per_class 100
```

This diffusion generation process is the most time-consuming process, so we added routine to use multiple GPUs to accelerate the process. To use multiple GPUs, simply set the environment variables. For example,

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m scripts.diffuse_img_attention --dataset_dir ./example_dataset --image_per_class 100
```

## Step 3: refine masks with few-shot conditioning

We use the following command to refine masks with few-shot conditioning.

```bash
python -m scripts.mask_generation --dataset_dir ./example_dataset --fss_ckpt_path pretrained_weights/voc_2_resnet101_5shot_78.89.pth
```

TODOs

- [x] Switch to modern diffusers
- [ ] Use xformers to speed up inversion / generation
- [ ] Integrate LLM / CLIPRetrieval to augment text prompts
- [ ] Switch to a better few-shot segmenter in mask generation (more recent networks & more pre-training data?)
- [ ] Integrate SAM to further refine mask generation?
- [ ] Add image / text template augmentations to the NM inversion to further stablize the training process.
- [ ] Make a prettier tqdm pbar during image-attention synthesis.
