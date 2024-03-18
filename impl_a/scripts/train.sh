coco_path=../../data/coco2017
num_gpus=8

EXP_DIR=exps/ms_detr_a_300

mkdir -p $EXP_DIR

GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python -u main.py \
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
   --o2m_cls_loss_coef 2 \
   > $EXP_DIR/train.log
