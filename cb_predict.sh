
TASK=CB
for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=0 python predict.py \
--task $TASK \
--output_dir ./outputs/cb/$SEED/ \
--data_dir CB-bin 
done