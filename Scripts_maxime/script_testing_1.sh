start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1
#~ ./examples/mult/sgemm -3d -xyz $((960*N)) -nblocks $((N)) -nblocksz $((N)) -iter 1
#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5 # Attention il faut enable max buffer pour ca avec plus de 5 en degrée max
#~ libtool --mode=execute gdb --args 
#~ /./home/gonthier/these_gonthier_maxime/Code/permutation_visu_python $((N)) ${ORDO} 1 NDIMENSIONS
#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} NIDMENSIONS
#~ 2>&1 | tee Output_maxime/terminal_output.txt
#~ make -j 6
#~ Quand on lance la visu python il faut PRINTF=1 PRINT_N=$((N))

#~ make -C src/ -j 6
#~ make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

N=30 #je suis censé avoir 12721.1 pour N=30 ou 12806.9 depuis la maj
#~ N=20
#~ N=5 
#~ N=65
N=40 # 12893.0
N=1000

NGPU=1
#~ NGPU=2
#~ NGPU=3

#~ ORDO="dynamic-data-aware" # DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION))
#~ ORDO="HFP" # BELADY=$((BELADY)) ORDER_U=1
#~ ORDO="dmdar"
ORDO="modular-eager-prefetching"
#~ ORDO="modular-heft"
#~ ORDO="eager"
#~ ORDO="cuthillmckee"

#~ CM=500
#~ CM=250

EVICTION=0
#~ EVICTION=1

READY=0
#~ READY=1

TH=10

CP=5

#~ HOST="gemini-1-ipdps"
#~ HOST="gemini-2-ipdps"
HOST="gemini-1-fgcs"
#~ HOST="attila"

SEED=1

PRINTF=0
#~ PRINTF=1

TRACE=0
#~ TRACE=1

BELADY=0
#~ BELADY=1

MULTI=0
#~ MULTI=1
#~ MULTI=2
#~ MULTI=3
#~ MULTI=4
#~ MULTI=6

STEALING=0
#~ STEALING=3

NITER=1
#~ NITER=11

#~ TAILLE_TUILE=960
TAILLE_TUILE=1920

#~ echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
#~ STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))

STARPU_SCHED=${ORDO} STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter 1

#~ FASTER_FIRST_ITERATION=0 PRINT_TIME=0 PRINT_N=$((N)) REVERSE=1 PRINT3D=0 RANDOM_TASK_ORDER=0 RANDOM_DATA_ACCESS=0 TASK_STEALING=$((STEALING)) STARPU_BUS_STATS=1 MULTIGPU=$((MULTI)) STARPU_GENERATE_TRACE=$((TRACE)) PRINTF=$((PRINTF)) SEED=$((SEED)) STARPU_SCHED=${ORDO} BELADY=$((BELADY)) ORDER_U=1 STARPU_SCHED_READY=$((READY)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1

#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ /./home/gonthier/these_gonthier_maxime/Code/permutation_visu_python $((N)) ${ORDO} 1 4
#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 4

#~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."

# Pour tester sur Grid5k
#~ STARPU_SCHED=HFP COUNT_DO_SCHEDULE=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*10)) -nblocks 10 -iter 11
#~ STARPU_SCHED=HFP COUNT_DO_SCHEDULE=0 STARPU_SCHED_READY=0 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 libtool --mode=execute gdb --args ./examples/mult/sgemm -xy $((960*50)) -nblocks 50 -iter 1

#~ STARPU_CALIBRATE=1 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*35)) -nblocks $((35)) -iter 11

#~ STARPU_CALIBRATE=1 RANDOM_TASK_ORDER=1 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks $((15)) -iter 11

#~ STARPU_CALIBRATE=0 STARPU_GENERATE_TRACE=1 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*35)) -nblocks $((35)) -nblocksz $((4)) -iter 2

#~ STARPU_CALIBRATE=0 STARPU_GENERATE_TRACE=1 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*35)) -nblocks $((35))
		    		    
