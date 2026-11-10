#!/usr/bin/env bash
# Public environment template; contains no account-specific credentials or endpoints.
export COREX_HOME=/usr/local/corex
export CUDA_HOME=/usr/local/corex
export PATH="$COREX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$COREX_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export MAX_JOBS=1
export USE_TF=0
export USE_FLAX=0
