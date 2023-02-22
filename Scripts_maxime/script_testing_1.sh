#~ export CFLAGS+="-DTOTO"
 #~ avant le configure

start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1
#~ ./examples/mult/sgemm -3d -xyz $((960*N)) -nblocks $((N)) -nblocksz $((N)) -iter 1
#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5 # Attention il faut enable max buffer pour ca avec plus de 5 en degrée max
#~ libtool --mode=execute gdb --args 
#~ /./home/gonthier/these_gonthier_maxime/Code/permutation_visu_python $((N)) ${ORDO} 1 NDIMENSIONS
#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} NIDMENSIONS + dans la commande avant : PRINT_IN_TERMINAL=1 PRINT3D=1 PRINT_N=$((N))
#~ 2>&1 | tee Output_maxime/terminal_output.txt
#~ make -j 6
#~ Quand on lance la visu python il faut PRINTF=1 PRINT_N=$((N))
#~ compiler avec --enable-debug puis dans gdb tools/gdbinit
#~ --cfg=contexts/factory:thread si j'ai un crash avec simgrid. A mettre après l'appli
#~ -bound
#~ -no-prio
#~ STARPU_WATCHDOG_TIMEOUT=1000000000 STARPU_WATCHDOG_CRASH=1 

make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

# TAILLE_TUILE=1920
#~ TAILLE_TUILE=2880
#~ TAILLE_TUILE=3840
#~ HOST="gemini-1-cho_dep"
#~ truncate -s 0 Output_maxime/tgflops.txt
#~ for ((j=1 ; j<=1; j++))
# for ((j=1 ; j<=4; j++))
#~ for ((j=2 ; j<=2; j++))
#~ for ((j=4 ; j<=4; j++))
#~ for ((j=8 ; j<=8; j++))
#~ do
	#~ if [ $((j)) == 1 ]
	#~ then
		#~ NGPU=1
	#~ elif [ $((j)) == 2 ]
	#~ then
		#~ NGPU=2
	#~ elif [ $((j)) == 3 ]
	#~ then
		#~ NGPU=4
	#~ elif [ $((j)) == 4 ]
	#~ then
		#~ NGPU = 8
	#~ fi
	#~ for ((i=1 ; i<=20; i++))
	#~ do
		#~ N=$((i*5))
		#~ echo "N = "${N} "- "${NGPU} "GPU - " ${TAILLE_TUILE}
		#~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((${TAILLE_TUILE}*N)) -nblocks $((N)) | tail -n 3 >> Output_maxime/tgflops.txt
	#~ done
#~ done
#~ exit

#~ N=3
#~ N=4
#~ N=5
#~ N=10
#~ N=15
#~ N=20
#~ N=25
#~ N=30
#~ N=35
#~ N=40
#~ N=45
#~ N=50
#~ N=55
N=60 # nb_data_looked_at between 1300 and 1500 each time
#~ N=65
#~ N=70

NGPU=1
#~ NGPU=2
#~ NGPU=3
#~ NGPU=4
#~ NGPU=8

ORDO="dynamic-data-aware"
#~ ORDO="dmdar"
#~ ORDO="lws"
#~ ORDO="graph_test" # STARPU_SCHED_GRAPH_TEST_DESCENDANTS= 0 ou 1 pour activer les descendants
#~ ORDO="dmdas"
#~ ORDO="modular-eager-prefetching"
#~ ORDO="modular-heft"
#~ ORDO="eager"
#~ ORDO="cuthillmckee"
#~ ORDO="HFP" # BELADY=$((BELADY)) ORDER_U=1

#~ CM=250
#~ CM=500
#~ CM=1000
CM=2000
#~ CM=0 # 0 = infinie
#~ CM=100

#~ EVICTION=0
EVICTION=1

#~ READY=0
READY=1

TH=10
#~ TH=0

CP=5 # CUDA_PIPELINE
#~ CP=0

#~ HOST="attila"
#~ HOST="gemini-1-ipdps"
#~ HOST="gemini-2-ipdps"
#~ HOST="gemini-1-fgcs"
#~ HOST="gemini-1-fgcs-36"
#~ HOST="gemini-1-cho_dep"
HOST="gemini-1-cho_dep_corrected"

SEED=1

PRINTF=0
#~ PRINTF=1
#~ PRINTF=2  # Pour Cholesky

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
#~ NITER=2
#~ NITER=11

#~ TAILLE_TUILE=960
TAILLE_TUILE=1920
#~ TAILLE_TUILE=2880
#~ TAILLE_TUILE=3840
#~ TAILLE_TUILE=4800
#~ Ne pas oublier : -z $((TAILLE_TUILE*4)) !!!

#~ APP3D=0
APP3D=1

TRACE=0
#~ TRACE=1

SPARSE=0
#~ SPARSE=10

#~ TASK_ORDER=0
#~ TASK_ORDER=1
TASK_ORDER=2

#~ DATA_ORDER=0
#~ DATA_ORDER=1
DATA_ORDER=2

#~ FREE_PUSHED_TASK_POSITION=0
FREE_PUSHED_TASK_POSITION=1

#~ PRIORITY_ATTRIBUTION=0
PRIORITY_ATTRIBUTION=1
#~ PRIORITY_ATTRIBUTION=2

GRAPH_DESCENDANTS=0
#~ GRAPH_DESCENDANTS=1
#~ GRAPH_DESCENDANTS=2

#~ DOPT_SELECTION_ORDER=0
#~ DOPT_SELECTION_ORDER=1
#~ DOPT_SELECTION_ORDER=2
#~ DOPT_SELECTION_ORDER=3 # Bug on this one mono gpu N55
#~ DOPT_SELECTION_ORDER=4
#~ DOPT_SELECTION_ORDER=5
#~ DOPT_SELECTION_ORDER=6
DOPT_SELECTION_ORDER=7

#~ HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=0
HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=1

SIMULATE_MEMORY=0
#~ SIMULATE_MEMORY=1

CHOOSE_BEST_DATA_FROM=0
#~ CHOOSE_BEST_DATA_FROM=1

CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET=0
#~ CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET=1

#~ PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=0
#~ PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=1
PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=2

#~ THRESHOLD=0
#~ THRESHOLD=1
THRESHOLD=2

APPLICATION="./examples/cholesky/cholesky_implicit -size $((${TAILLE_TUILE}*N)) -nblocks $((N))"
#~ APPLICATION="./examples/cholesky/cholesky_implicit -size $((${TAILLE_TUILE}*N)) -nblocks $((N)) -check"
#~ APPLICATION="libtool --mode=execute gdb --args ./examples/cholesky/cholesky_implicit -size $((TAILLE_TUILE*N)) -nblocks $((N)) --cfg=contexts/factory:thread"
#~ APPLICATION="./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) -no-prio"
#~ APPLICATION="./examples/cholesky/cholesky_implicit -size $((${TAILLE_TUILE}*N)) -nblocks $((N)) -bound"
#~ APPLICATION="./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 11"

echo -e "\nN=${N} - NGPU=${NGPU} - TAILLE_TUILE=${TAILLE_TUILE} - SCHEDULER=${ORDO} - STARPU_NTASKS_THRESHOLD=${TH} - STARPU_CUDA_PIPELINE=${CP} - HOST=${HOST} - APPLICATION=${APPLICATION}\n"

# DMDAR - Cholesky avec dépendances
#~ PRIORITY_ATTRIBUTION=$((PRIORITY_ATTRIBUTION)) STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} STARPU_SCHED=${ORDO} STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ${APPLICATION}

# DARTS - Cholesky avec dépendances
THRESHOLD=$((THRESHOLD)) PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=$((PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK)) CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET=$((CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET)) CHOOSE_BEST_DATA_FROM=$((CHOOSE_BEST_DATA_FROM)) SIMULATE_MEMORY=$((SIMULATE_MEMORY)) DOPT_SELECTION_ORDER=$((DOPT_SELECTION_ORDER)) HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=$((HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE)) GRAPH_DESCENDANTS=$((GRAPH_DESCENDANTS)) PRIORITY_ATTRIBUTION=$((PRIORITY_ATTRIBUTION)) FREE_PUSHED_TASK_POSITION=$((FREE_PUSHED_TASK_POSITION)) STARPU_WORKER_STATS=0 DATA_ORDER=$((DATA_ORDER)) TASK_ORDER=$((TASK_ORDER)) DEPENDANCES=1 PRIO=1 APP=$((APP3D)) STARPU_GENERATE_TRACE=$((TRACE)) SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED_READY=$((READY)) DATA_POP_POLICY=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=0 STARPU_HOSTNAME=${HOST} STARPU_SCHED=${ORDO} STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=0 ${APPLICATION} 2>&1 | tee Output_maxime/terminal_output.txt

# Sur Grid5k
#~ N=40 ; TAILLE_TUILE=1920 ; NGPU=4 ; DOPT_SELECTION_ORDER=1 PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=1 CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET=0 PRIORITY_ATTRIBUTION=1 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 STARPU_LIMIT_CUDA_MEM=$((2000)) HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=1 GRAPH_DESCENDANTS=0 STARPU_SCHED_READY=1 TASK_ORDER=2 DATA_ORDER=2 FREE_PUSHED_TASK_POSITION=1 DEPENDANCES=1 PRIO=1 APP=1 SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((10)) STARPU_CUDA_PIPELINE=$((5)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 libtool --mode=execute gdb --args ./examples/cholesky/cholesky_implicit -size $((TAILLE_TUILE*N)) -nblocks $((N))

#~ N=40 ; TAILLE_TUILE=1920 ; NGPU=4 ; DOPT_SELECTION_ORDER=1 PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=1 CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET=0 PRIORITY_ATTRIBUTION=1 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 STARPU_LIMIT_CUDA_MEM=$((2000)) HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=1 GRAPH_DESCENDANTS=0 STARPU_SCHED_READY=1 TASK_ORDER=2 DATA_ORDER=2 FREE_PUSHED_TASK_POSITION=1 DEPENDANCES=1 PRIO=1 APP=1 SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((10)) STARPU_CUDA_PIPELINE=$((5)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 libtool --mode=execute gdb --args ./examples/cholesky/cholesky_implicit -size $((TAILLE_TUILE*N)) -nblocks $((N)) > Output_maxime/terminal_output.txt

# HFP
#~ DEPENDANCES=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
