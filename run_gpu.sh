#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python lstm_ucsc_gpu.py output/ 300 $1 $2 16 >> ../logs/log_$3_$1_$2

