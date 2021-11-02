start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1
#~ ./examples/mult/sgemm -3d -xyz $((960*N)) -nblocks $((N)) -nblocksz $((N)) -iter 1
#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5
#~ libtool --mode=execute gdb --args 
#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 1
#~ setarch linux64 -R libtool --mode=execute gdb --args
#~ 2>&1 | tee Output_maxime/terminal_output.txt
#~ make -j 6
#~ Quand on lance la visu python il faut PRINTF=1 PRINT_N=$((N))

#~ make -C src/ -j 6
make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

N=30 #je suis censé avoir 12721.1 pour N=30
#~ N=20
#~ N=5 
#~ N=65
N=30

NGPU=1
#~ NGPU=2

#~ ORDO="dynamic-data-aware" # DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION))
ORDO="HFP" # BELADY=$((BELADY)) ORDER_U=1
#~ ORDO="dmdar"

CM=500
#~ CM=250

EVICTION=0
#~ EVICTION=1

READY=0
#~ READY=1

TH=10

CP=5

HOST="gemini-1-ipdps"
#~ HOST="gemini-2-ipdps"
#~ HOST="attila"

SEED=1

PRINTF=0
#~ PRINTF=1

TRACE=0
#~ TRACE=1

#~ BELADY=0
BELADY=1

MULTI=0
#~ MULTI=1
#~ MULTI=2
#~ MULTI=3
#~ MULTI=4
#~ MULTI=6

STEALING=0
#~ STEALING=3

NITER=1
NITER=11

PRINT_N=$((N)) TASK_STEALING=$((STEALING)) STARPU_BUS_STATS=1 MULTIGPU=$((MULTI)) STARPU_GENERATE_TRACE=$((TRACE)) PRINTF=$((PRINTF)) SEED=$((SEED)) STARPU_SCHED=${ORDO} BELADY=$((BELADY)) ORDER_U=1 STARPU_SCHED_READY=$((READY)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER))
#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 1

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."

# Pour tester sur Grid5k
#~ STARPU_SCHED=HFP COUNT_DO_SCHEDULE=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*10)) -nblocks 10 -iter 1
		    
		    
