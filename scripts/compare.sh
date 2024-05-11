#!/bin/bash

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts

# $HOME_DIR/env/bin/python3 $SCRIPT_DIR/compare_mm.py --dtype 32 --b 32 --n 24 --m 224 --p 224

# $HOME_DIR/env/bin/python3 $SCRIPT_DIR/compare_mm.py --dtype 32 --b 64 --n 224 --m 448 --p 448

# $HOME_DIR/env/bin/python3 $SCRIPT_DIR/compare_sdpa.py --dtype 32 --b 64 --h 12 --s_q 64 --s_kv 64 --d_qk 32 --d_v 768

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/compare_conv.py --dtype 32 --b 32 --in_channels 64 --iH 224 --iW 224 --out_channels 64 --groups 1 --kH 3 --kW 3 --stride 1 --padding 0 --dilation 1