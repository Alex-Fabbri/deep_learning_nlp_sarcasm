#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python bi-lstm_ucsc_gpu.py output/ 300 False False 16 >> logs/log_bi_$1

