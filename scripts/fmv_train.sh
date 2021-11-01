python3 tools/train.py \
-f exps/fmv/yolox_x_ablation_fmv.py \
--devices 8 --batch-size 64 \
--fp16 --occupy \
--experiment-name fmv_exp_16_giou \
--ckpt pretrained/bytetrack_x_mot17.pth.tar \
--max_iter 40 \
motion True tsm True