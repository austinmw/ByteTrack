EXP_NAME=$1

python3 tools/train.py \
-f exps/visdrone/yolox_x_ablation_visdrone.py \
--devices 8 --batch-size 48 \
--fp16 --occupy \
--experiment-name $EXP_NAME \
--ckpt pretrained/yolox_x.pth \
motion True tsm True
