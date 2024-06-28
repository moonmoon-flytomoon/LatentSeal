# The Latent Seal: Robust Watermarking for Latent Diffusion Model
This repo is the official code for
* **The Latent Seal: Robust Watermarking for Latent Diffusion Model**

## Current Version
* Test Demo release.
* Model release.
* Latent Seal Dataset release.
* Training code will be released when this paper is accepted.


### Test Demo
- Here we provide a [Test Demo](https://www.kaggle.com/code/kevin1742064161/notebook/notebook) deployed on Kaggle.
### Model
- Latent Seal traning log  is publically available at [[WandB]](https://api.wandb.ai/links/moonmoon-flytomoon/bvi297g2).
- Latent Seal model weight  is publically available at [[Kaggle]](https://www.kaggle.com/datasets/kevin1742064161/code-weight/data).
### Dataset
- In this paper, we introduce the Latent Seal Dataset.
The dataset is publically available at [[HuggingFace]](https://huggingface.co/datasets/moonmoon-Flytomoon/LSD).

## Requirements
This codebase has been developed with python version 3.10, PyTorch version 2.1.0, CUDA 12.1.

First, clone the repository locally and move inside the folder:
```cmd
git clone https://github.com/moonmoon-flytomoon/LatentSeal
cd LatentSeal
```
Next, create a new virtual environment with anaconda:
```cmd
conda create -n LatentSeal python=3.10
conda activate LatentSeal
```

Next, install [PyTorch](https://pytorch.org/):
```cmd
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies with pip:
```cmd
pip install -r requirements.txt
```

## Get Started
- Run `python src/train.py` for training.

- Run `python Test/test.py` for testing.
