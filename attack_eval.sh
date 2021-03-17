#!/bin/sh
#
#SBATCH -G 1

MODEL_PATH=/data/private/chenyingfa/rbt-sst-tmixada-pwws-iterative/final-checkpoint
DATA_DIR=/home/chenyingfa/MixADA/data/MixADA-data/ag-news/test.tsv
SEQLEN=128
TASK=ag-news


python3 attackEval.py  \
--model_name_or_path ${MODEL_PATH} \
--model_type roberta \
--attacker pwws \
--data_dir ${DATA_DIR} \
--max_seq_len $SEQLEN \
--task_name ${TASK}