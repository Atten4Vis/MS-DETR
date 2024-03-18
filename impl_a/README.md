# MS-DETR

This is the PyTorch implementation of MS-DETR of **implementation (a)** in the paper "[MS-DETR: Efficient DETR Training with Mixed Supervision](https://arxiv.org/pdf/2401.03989.pdf)"

<div align=center>  
<img src='../assets/ms-detr-implementations.png' width="100%">
</div>


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

Train MS-DETR based on Deformable DETR with 8 GPUs.

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
   --cls_loss_coef 1 \
   --o2m_cls_loss_coef 2
```

### Evaluation

Evaluate MS-DETR with 8 GPUs.

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
