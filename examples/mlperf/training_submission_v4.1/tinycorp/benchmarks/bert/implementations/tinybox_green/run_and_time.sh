#!/bin/bash

export PYTHONPATH="."
export MODEL="bert"
export SUBMISSION_PLATFORM="tinybox_green"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=54 EVAL_BS=6

export BEAM=4
export IGNORE_JIT_FIRST_BEAM=1
export BASEDIR="/raid/datasets/wiki"

# pip install -e ".[mlperf]"
export LOGMLPERF=1

export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
LOGFILE="bert_green_${DATETIME}_${SEED}.log"

# init
BENCHMARK=10 INITMLPERF=1 python3 examples/mlperf/model_train.py | tee $LOGFILE

# run
RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a $LOGFILE
