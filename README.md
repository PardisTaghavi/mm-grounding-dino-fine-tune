This is just a copy of original repo (https://github.com/open-mmlab/mmdetection/tree/main) for personal fine-tuning purposes. Please refer to the original library,




Fine-tuning Few-shot mm-Gdino on Cityscapes Dataset.

1) Installation: https://mmdetection.readthedocs.io/en/latest/get_started.html

2) Dataset: https://drive.google.com/file/d/143yo4N2guTVst_824xehFqgdHSZAHQbr/view?usp=sharing
3) Pretrained model: https://drive.google.com/file/d/1Q_DEBxPzcSqOpXvzSpIX0pv68wurrG05/view?usp=sharing

  
5) change data_root in configs/mm_grounding_dino/cityscapes/grounding_dino_swin-l_finetune_cityscapes_186_fewshot_pretrain_all.py
6) change "load_from" path to the pretrined model path.

7) ./tools/dist_train.sh configs/mm_grounding_dino/cityscapes/grounding_dino_swin-l_finetune_cityscapes_186_fewshot_pretrain_all.py 2


