# MS-DETR

This repository is the official implementation of "[MS-DETR: Efficient DETR Training with Mixed Supervision](https://arxiv.org/pdf/2401.03989.pdf)"

Authors: Chuyang Zhao, Yifan Sun, Wenhao Wang, Qiang Chen, Errui Ding, Yi Yang, Jingdong Wang


**Implementations**

- The Paddle version of MS-DETR will be available in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

- The PyTorch implementation of MS-DETR architecture (a) (Figure 4 in main paper) is available [here](./impl_a/).

## Introduction

DETR accomplishes end-to-end object detection through iteratively generating multiple object candidates based on image features and promoting one candidate for each ground-truth object. The traditional training procedure using one-to-one supervision in the original DETR lacks direct supervision for the object detection candidates.

<div align=center>
<img src='assets/ms-detr-illustration.png' width="48%">
<img src='assets/ms-detr-loss-curve.png' width="48%">
</div>

We aim at improving the DETR training efficiency by explicitly supervising the candidate generation procedure through mixing one-to-one supervision and one-to-many supervision. Our approach, namely MS-DETR, is simple, and places one-to-many supervision to the object queries of the primary decoder that is used for inference. In comparison to existing DETR variants with one-to-many supervision, such as Group DETR and Hybrid DETR, our approach does not need additional decoder branches or object queries. The object queries of the primary decoder in our approach directly benefit from one-to-many supervision and thus are superior in object candidate prediction. Experimental results show that our approach outperforms related DETR variants, such as DN-DETR, Hybrid DETR, and Group DETR, and the combination with related DETR variants further improves the performance.

<div align=center>  
<img src='assets/ms-detr-implementations.png' width="100%">
</div>


## Model Zoo

We provide the checkpoint of the following models:

| Name                                         | Baseline          | Backbone | Queries | Epochs | mAP  | Download  |
| -------------------------------------------- | ----------------- | -------- | ------- | ------ | ---- | --------- |
| [MS-DETR](./scripts/ms_detr_ddetr_300.sh)    | Deformable-DETR   | R50      | 300     | 12     | 47.6 | [model](https://drive.google.com/drive/folders/1DIwQfvnBpkLAbIT8MrHgH3NBy0XlWKq4) |
| [MS-DETR](./scripts/ms_detr_ddetr_pp_300.sh) | Deformable-DETR++ | R50      | 300     | 12     | 48.8 | [model](https://drive.google.com/drive/folders/1DIwQfvnBpkLAbIT8MrHgH3NBy0XlWKq4) |
| [MS-DETR](./scripts/ms_detr_ddetr_pp_900.sh) | Deformable-DETR++ | R50      | 900     | 12     | 50.0 | [model](https://drive.google.com/drive/folders/1DIwQfvnBpkLAbIT8MrHgH3NBy0XlWKq4) |

## Installation

We tested our code with `Python=3.10, PyTorch=1.12.1, CUDA=11.3`. Please install PyTorch first according to [official instructions](https://pytorch.org/get-started/previous-versions/).

1. Clone the repository.

```sh
git clone https://github.com/Atten4Vis/MS-DETR.git
cd MS-DETR
```

2. Install dependencies.

```sh
pip install -r requirements.txt
```

3. Compile MSDeformAttn CUDA operators.

```sh
cd models/ops
python setup.py build install
```

## Data

We use the COCO-2017 dataset for training and evaluation. Please [download](https://cocodataset.org/) and organize the dataset as follows:

```
coco_path/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

## Run

### Training

Train MS-DETR with 8 GPUs based on Deformable-DETR:

```sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 python -u main.py \
   --output_dir $EXP_DIR \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --epochs 12 \
   --lr_drop 11 \
   --coco_path=$coco_path \
   --num_queries 300 \
   --use_ms_detr \
   --use_aux_ffn \
   --cls_loss_coef 1 \
   --o2m_cls_loss_coef 2
```

Other training scripts are available in [./scripts](./scripts).

### Evaluation

Evaluate MS-DETR with 8 GPUs:

```sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 python -u main.py \
    --coco_path=$coco_path \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries 300 \
    --use_ms_detr \
    --use_aux_ffn \
    --resume $EXP_DIR/checkpoint.pth \
    --eval
```

## Acknowledgement
Our code is based on [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), [H-DETR](https://github.com/HDETR/H-Deformable-DETR) and [DETA](https://github.com/jozhang97/DETA). Thanks for their great works.

## Citation

If you use MS-DETR in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@article{zhao2024ms,
  title={MS-DETR: Efficient DETR Training with Mixed Supervision},
  author={Zhao, Chuyang and Sun, Yifan and Wang, Wenhao and Chen, Qiang and Ding, Errui and Yang, Yi and Wang, Jingdong},
  journal={arXiv preprint arXiv:2401.03989},
  year={2024}
}
```
