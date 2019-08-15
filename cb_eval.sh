TOTAL_NUM_UPDATES=120  # 15 epochs through RTE for bsz 16
EPOCHS=15
WARMUP_UPDATES=6      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=3
MAX_SENTENCES=32        # Batch size.

# example_per_epoch =250  batch_size =32  batch_per_epoch = 8  
for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=3 python eval.py CB-bin/ \
--seed $SEED \
--no-shuffle \
--no-epoch-checkpoints \
--restore-file ../../../outputs/cb/$SEED/checkpoint_best.pt  \
--save-dir ./outputs/cb/$SEED \
--max-positions 512 \
--max-sentences $MAX_SENTENCES \
--max-tokens 4400 \
--task sentence_prediction \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--init-token 0 --separator-token 2 \
--arch roberta_large \
--criterion sentence_prediction \
--num-classes $NUM_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
--max-epoch $EPOCHS \
--find-unused-parameters \
--best-checkpoint-metric acc_f1_avg --maximize-best-checkpoint-metric;

done