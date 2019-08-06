TOTAL_NUM_UPDATES=2036  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=122      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size.

# example_per_epoch =2490  batch_size =16  batch_per_epoch = 156  
CUDA_VISIBLE_DEVICES=0 python train.py RTE-bin/ \
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
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;

# bash ./examples/roberta/preprocess_SuperGLUE_tasks.sh  ../data-superglue-csv RTE
# bash ./examples/roberta/preprocess_GLUE_tasks.sh  ../glue_data RTE