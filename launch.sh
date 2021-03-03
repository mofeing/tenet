#!/bin/bash
export PYTHONPATH=/home/bsc21/bsc21106/tenet/test:/home/bsc21/bsc21106/tenet/src:$PYTHONPATH
enqueue_compss --num_nodes=1 --qos=debug --log_level=debug --exec_time=5 --summary --graph $@
