## Overview
This is just a copy of original repo (https://github.com/open-mmlab/mmdetection/tree/main) for personal fine-tuning purposes. Please refer to the original library for more details.

## Fine-tuning Few-shot mm-Gdino on Cityscapes Dataset

## Installation

Based on https://mmdetection.readthedocs.io/en/latest/get_started.html

```bash
apt-get update && apt-get install -y emacs-nox python3-pip

pip install gdown openmim
pip install torch torchvision
mim install mmengine 
mim install "mmcv>=2.0.0"
mim install mmdet
```

## Artifacts

Use gdown to download the artifacts.

Few-shot Labeled Data

Dataset: https://drive.google.com/file/d/143yo4N2guTVst_824xehFqgdHSZAHQbr/view?usp=sharing
Pretrained model: https://drive.google.com/file/d/1Q_DEBxPzcSqOpXvzSpIX0pv68wurrG05/view?usp=sharing

```bash
gdown https://drive.google.com/uc?id=143yo4N2guTVst_824xehFqgdHSZAHQbr
gdown https://drive.google.com/uc?id=1Q_DEBxPzcSqOpXvzSpIX0pv68wurrG05
```

Few-shot Labeled Data + Unlabeled Data [train THIS!]

```bash
gdown https://drive.google.com/uc?id=1trqXGRW9aSSVeZUTFFdNP1lYj-leD9Od
gdown https://drive.google.com/uc?id=1Q_DEBxPzcSqOpXvzSpIX0pv68wurrG05
```

## Usage  
In configs/mm_grounding_dino/cityscapes/grounding_dino_swin-l_finetune_cityscapes_186_fewshot_pretrain_all.py,

1) change "data_root" to the dataset path.
2) change "load_from" to the pretrained model path.

```bash
./tools/dist_train.sh configs/mm_grounding_dino/cityscapes/grounding_dino_swin-l_finetune_cityscapes_186_fewshot_pretrain_all.py 2
```
## Result - Few shot labeled data (10imgs/cls)

| LR/Scheduler   | MultiStep(weight decay0.01)     | 
|----------------|---------------------------------|
| 5e-5           |   AP_{bbox} 53.1                |
| 1e-5           |   AP_{bbox} 51.5                | 

## Result - Few shot labeled data (10imgs/cls) + unlabeled data(1860 imgs) for self-training

| LR/Scheduler   | MultiStep(weight decay0.01)     | 
|----------------|---------------------------------|
| 5e-5           |   AP_{bbox} 54.20               |

