
TASK=RTE
for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=2 python eval2.py \
--task $TASK \
--output_dir ./outputs/rte/$SEED/ \
--data_dir RTE-bin
done