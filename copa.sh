TOTAL_NUM_UPDATES=500  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=30      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size.

# example_per_epoch =400*2  batch_size =16  batch_per_epoch = 25*2
CUDA_VISIBLE_DEVICES=3 python train.py COPA-bin/ \
--no-shuffle \
--restore-file '../roberta.large/swag/checkpoint_best.pt' \
--save-dir '../roberta.large/' \
--max-positions 512 \
--max-sentences $MAX_SENTENCES \
--max-tokens 4400 \
--task multiple_choice \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--init-token 0 --separator-token 2 \
--arch roberta_large \
--criterion multiple_choice \
--num-classes $NUM_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
--max-epoch 10 \
--find-unused-parameters \
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;