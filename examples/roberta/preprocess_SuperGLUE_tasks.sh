#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# raw glue data as downloaded by glue download script (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
if [[ $# -ne 2 ]]; then
  echo "Run as following:"
  echo "./examples/roberta/preprocess_GLUE_tasks.sh <glud_data_folder> <task_name>"
  exit 1
fi

GLUE_DATA_FOLDER=$1

# download bpe encoder.json, vocabulary and fairseq dictionary
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASKS=$2

if [ "$TASKS" = "ALL" ]
then
  TASKS="COPA CB RTE"
fi

for TASK in $TASKS
do
  echo "Preprocessing $TASK"

  TASK_DATA_FOLDER="$GLUE_DATA_FOLDER/$TASK"
  echo "Raw data as downloaded from superglue website: $TASK_DATA_FOLDER"

  SPLITS="train val test"
  INPUT_COUNT=2
  if [ "$TASK" = "COPA" ]
  then
    INPUT_COLUMNS=( 1 2 )
    TEST_INPUT_COLUMNS=( 1 2 )
    LABEL_COLUMN=3
  elif [ "$TASK" = "CB" ]
  then
    INPUT_COLUMNS=( 1 2 )
    TEST_INPUT_COLUMNS=( 1 2 )
    LABEL_COLUMN=3
  elif [ "$TASK" = "RTE" ]
  then
    INPUT_COLUMNS=( 1 2 )
    TEST_INPUT_COLUMNS=( 1 2 )
    LABEL_COLUMN=3
  elif [ "$TASK" = "SWAG" ]
  then
    INPUT_COLUMNS=( 1 2 )
    TEST_INPUT_COLUMNS=( 1 2 )
    LABEL_COLUMN=3
  fi

  # Strip out header and filter lines that don't have expected number of fields.
  rm -rf "$TASK_DATA_FOLDER/processed"
  mkdir "$TASK_DATA_FOLDER/processed"
  for SPLIT in $SPLITS
  do
    tail -n +2 "$TASK_DATA_FOLDER/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
    cp "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv";
    rm "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
  done

  # Split into input0, input1 and label
  for SPLIT in $SPLITS
  do
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
      if [[ "$SPLIT" != test* ]]
      then
        COLUMN_NUMBER=${INPUT_COLUMNS[$INPUT_TYPE]}
      else
        COLUMN_NUMBER=${TEST_INPUT_COLUMNS[$INPUT_TYPE]}
      fi
      cut -f"$COLUMN_NUMBER" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.raw.input$INPUT_TYPE";
    done

    if [[ "$SPLIT" != test* ]]
    then
      cut -f"$LABEL_COLUMN" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.label";
    fi

    # BPE encode.
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
      LANG="input$INPUT_TYPE"
      echo "BPE encoding $SPLIT/$LANG"
      python -m examples.roberta.multiprocessing_bpe_encoder \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$TASK_DATA_FOLDER/processed/$SPLIT.raw.$LANG" \
      --outputs "$TASK_DATA_FOLDER/processed/$SPLIT.$LANG" \
      --workers 60 \
      --keep-empty;
    done
  done

  # Remove output directory.
  rm -rf "$TASK-bin"

  DEVPREF="$TASK_DATA_FOLDER/processed/val.LANG"
  TESTPREF="$TASK_DATA_FOLDER/processed/test.LANG"

  # Run fairseq preprocessing:
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
  do
    LANG="input$INPUT_TYPE"
    fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.$LANG" \
      --validpref "${DEVPREF//LANG/$LANG}" \
      --testpref "${TESTPREF//LANG/$LANG}" \
      --destdir "$TASK-bin/$LANG" \
      --workers 60 \
      --srcdict dict.txt;
  done

  fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
      --validpref "${DEVPREF//LANG/'label'}" \
      --destdir "$TASK-bin/label" \
      --workers 60;
done
