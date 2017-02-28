#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python lstm_ucsc_context_bi_gpu.py output/ 300 False False 16 >> logs/log_bi_$1

