coco_path=../data/coco2017
num_gpus=8

EXP_DIR=exps/ms_detr_pp_900

GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python -u main.py \
   --output_dir $EXP_DIR \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --epochs 12 \
   --lr_drop 11 \
   --coco_path=$coco_path \
   --num_queries 900 \
   --dropout 0.0 \
   --mixed_selection \
   --look_forward_twice \
   --use_ms_detr \
   --use_aux_ffn \
   --topk_eval 300 \
   --resume $EXP_DIR/checkpoint.pth \
   --eval
