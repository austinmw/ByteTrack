python3 tools/train.py \
-f exps/visdrone/yolox_x_ablation_visdrone.py \
--devices 8 --batch-size 48 \
--fp16 --occupy \
--experiment-name vd_exp_01_motion \
--ckpt pretrained/yolox_x.pth \
motion True mosaic_dataset True
