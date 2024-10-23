# aligndiff_private

## Setup dependency

Clone the codebase.

```bash
cd ~
git clone --recursive https://github.com/RogerQi/aligndiff
```

Follow [INSTALL.md](./docs/INSTALL.md) to install required dependencies.

Also, some Stable Diffusion repos on Hugging face requires credentials. After the installation
is completed, `huggingface-cli` will be installed. Run the following command to login.

```bash
huggingface-cli login
```

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
python -m scripts.image_mask_inversion --dataset_dir ./example
```

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
python -m scripts.mask_generation --dataset_dir ./example_dataset
```
