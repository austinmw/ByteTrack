python3 tools/train.py \
-f exps/visdrone/yolox_x_ablation_visdrone.py \
-d 8 -b 32 --fp16 -o \
-c pretrained/yolox_x.pth \
--experiment-name visdrone_test_1 \
--max_iter 40