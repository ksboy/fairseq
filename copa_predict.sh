
TASK=COPA
for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=3 python predict_copa.py \
--task $TASK \
--output_dir ./outputs/copa/$SEED/ \
--data_dir COPA-bin
done