#!/usr/bin/bash
#~ bash Scripts_maxime/script_testing_1.sh

start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 50000000
#~ make -C src/ -j 6
make -j 6

#~ libtool --mode=execute strace ./examples/mult/sgemm

#~ libtool --mode=execute strace -o /tmp/log ./examples/mult/sgemm

#~ srun --exclusive -C sirocco21 --pty bash Scripts_maxime/task_stealing.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne task_stealing -i

#~ bash Scripts_maxime/task_stealing.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne task_stealing

#~ ./examples/mult/sgemm -3d -xy $((960*i)) -nblocks $((N)) -nblocksz $((4)) -iter 1
#~ ./examples/mult/sgemm -3d -xyz $((960*N)) -nblocks $((N)) -nblocksz $((N)) -iter 1
#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5

#~ libtool --mode=execute gdb --args 
#~ r backtrace l p

#~ HOST="attila"
#~ ORDO="dynamic-outer"
#~ N=7
#~ CUDAMEM=500
#~ NGPU=1
#~ BW=350

#~ SEED=1 STARPU_SCHED=${ORDO} EVICTION_STRATEGY_DYNAMIC_OUTER=0 DATA_POP_POLICY=1 PRINTF=1 PRINT_N=$((N)) STARPU_GENERATE_TRACE=0 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CUDAMEM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 

#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 1

N=45
NGPU=1
#~ ORDO="dynamic-outer"
ORDO="modular-eager-prefetching"
BW=350
CM=500
EVICTION=1
POP_POLICY=1
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 5000000

STARPU_SCHED=${ORDO} SEED=1 EVICTION_STRATEGY_DYNAMIC_OUTER=$((EVICTION)) DATA_POP_POLICY=$((POP_POLICY)) STARPU_GENERATE_TRACE=1 PRINTF=1 PRINT_N=$((N)) STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 
#~ 2>&1 | tee Output_maxime/terminal_output.txt

#~ SomeCommand 2>&1 | tee SomeFile.txt

#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 1

end=`date +%s` runtime=$((end-start)) echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
