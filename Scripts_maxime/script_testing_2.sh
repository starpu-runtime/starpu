#!/usr/bin/bash
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 50000000

STARPU_SCHED=HFP ORDER_U=1 ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1

echo "Fin du script"
