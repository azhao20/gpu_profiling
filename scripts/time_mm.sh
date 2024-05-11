#!/bin/bash


HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts
FINAL_CSV=$HOME_DIR/gpu_profiling/data/mm.csv

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/mm.py --mode "time" --dtype 16 --n 64 --m 224 --p 224 --out_file $FINAL_CSV

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/bmm.py --mode "time" --dtype 16 --b 64 --n 64 --m 224 --p 224 --out_file $FINAL_CSV

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/sdpa.py --mode "time" --dtype 32 --b 64 --h 12 --s_q 64 --s_kv 64 --d_qk 32 --d_v 768 --out_file $FINAL_CSV