## Setup dependencies

### Install Anaconda

Follow the instruction to install anaconda [here](https://www.anaconda.com/download).

Follow the instruction [here](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) to set up Mamba, a fast environment solver for conda.

```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Note: technically, the mamba solver should behave the same as the default solver. However, there have been cases where dependencies
can not be properly set up with the default mamba solver. The following instructions have **only** been tested on mamba solver.

### Setup python dependencies

```bash
conda create -y -n aligndiff python=3.8 && conda activate aligndiff
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
