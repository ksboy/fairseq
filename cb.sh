TOTAL_NUM_UPDATES=200  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=12      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=3
MAX_SENTENCES=16        # Batch size.

# example_per_epoch =250  batch_size =16  batch_per_epoch = 16  
CUDA_VISIBLE_DEVICES=0 python train.py CB-bin/ \
--no-shuffle \
--restore-file '../roberta.large.mnli/model.pt' \
--save-dir '../roberta.large.mnli/' \
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
--max-epoch 10 \
--best-checkpoint-metric f1 --maximize-best-checkpoint-metric;