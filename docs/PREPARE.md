## Setup huggingface

Many Stable Diffusion repos on Hugging face requires credentials. After the installation
is completed, `huggingface-cli` will be installed. Run the following command to login.

```bash
huggingface-cli login
```
## Download pre-trained weights

AlignDiff is agnostic of FSS, but we use [SSP Few-shot segmentation](https://github.com/fanq15/SSP).

In particular, we use the [weights](https://drive.google.com/file/d/1zUSXihIX2K8vqpTbQbu1XgXCqTLgb8w5/view?usp=sharing) pre-trained on Pascal-5-2 dataset split.
Note that this weight has not seen the dog or the albatross class in the example dataset.

Download this weight and put it at a path of your choice (e.g., `./pretrained_weights/voc_2_resnet101_5shot_78.89.pth`).
