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

#~ make -C src/ -j 6
make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

N=5 # 240.5 GF normalement
N=11 # 1162 GF normalement
N=20	
#~ N=25
#~ N=30 # 2805 GF normalement
#~ N=35
#~ N=45

NGPU=1
#~ NGPU=2
#~ NGPU=3
#~ NGPU=4

ORDO="dynamic-data-aware" # EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION))
#~ ORDO="dmdar"
#~ ORDO="modular-eager-prefetching"
#~ ORDO="modular-heft"
#~ ORDO="eager"
#~ ORDO="cuthillmckee"
#~ ORDO="HFP" # BELADY=$((BELADY)) ORDER_U=1

CM=500
#~ CM=0 # 0 = infinie
#~ CM=100

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

TAILLE_TUILE=960
#~ TAILLE_TUILE=1920
#~ TAILLE_TUILE=2880
#~ TAILLE_TUILE=3840
#~ TAILLE_TUILE=4800
#~ Ne pas oublier : -z $((TAILLE_TUILE*4)) !!!

APP3D=0
#~ APP3D=1

SPARSE=0
#~ SPARSE=10

DEPENDANCES=0
DEPENDANCES=1

TASK_ORDER=0
#~ TASK_ORDER=1
#~ TASK_ORDER=2

#~ DATA_ORDER=0
#~ DATA_ORDER=1
DATA_ORDER=2

#~ CHOOSE_BEST_DATA_FROM=0 STARPU_WORKER_STATS=1 MULTIGPU=$((MULTI)) REVERSE=1 FASTER_FIRST_ITERATION=0 RANDOM_TASK_ORDER=0 STARPU_SCHED_READY=$((READY)) BELADY=$((BELADY)) ORDER_U=1 SIMULATE_MEMORY=0 PRINT3D=0 PRINT_N=$((N)) STARPU_GENERATE_TRACE=0 APP=$((APP3D)) STARPU_SCHED=${ORDO} STARPU_HOSTNAME=${HOST} EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER))
#~ CHOOSE_BEST_DATA_FROM=0 MULTIGPU=$((MULTI)) REVERSE=1 FASTER_FIRST_ITERATION=0 RANDOM_TASK_ORDER=0 STARPU_SCHED_READY=$((READY)) BELADY=$((BELADY)) ORDER_U=1 SIMULATE_MEMORY=0 PRINT3D=0 PRINT_N=$((N)) STARPU_GENERATE_TRACE=0 APP=$((APP3D)) STARPU_SCHED=${ORDO} STARPU_HOSTNAME=${HOST} EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER))
#~ TASK_ORDER=$((TASK_ORDER)) DATA_ORDER=$((DATA_ORDER)) STARPU_WORKER_STATS=1 DEPENDANCES=$((DEPENDANCES)) CHOOSE_BEST_DATA_FROM=0 STARPU_SCHED_READY=$((READY)) SIMULATE_MEMORY=0 PRINT3D=0 STARPU_GENERATE_TRACE=0 APP=$((APP3D)) STARPU_SCHED=${ORDO} STARPU_HOSTNAME=${HOST} EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ 2>&1 | tee Output_maxime/terminal_output.txt
TASK_ORDER=$((TASK_ORDER)) DATA_ORDER=$((DATA_ORDER)) PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) 
#~ libtool --mode=execute gdb --args ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) 
#~ 2>&1 | tee Output_maxime/terminal_output.txt
#~ libtool --mode=execute gdb --args ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) STARPU_WATCHDOG_TIMEOUT=10000000 STARPU_WATCHDOG_CRASH=1 
#~ Compiler avec --enable-debug

#~ NGPU=3
#~ N=10
#~ echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
#~ STARPU_SCHED_READY=1 STARPU_BUS_STATS=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=HFP HMETIS_N=$((N)) HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1

#~ python3 /home/gonthier/these_gonthier_maxime/Code/Barplot_DARTS.py Output_maxime/DARTS_data_choosen_stats.csv
#~ mv Output_maxime/DARTS_data_choosen_stats.csv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/${DOSSIER}/DARTS_data_choosen_stats_N${N}_SIMMEM${SIMMEM}_FROMMEM${FROMMEM}.csv
#~ mv plot.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/${DOSSIER}/DARTS_data_choosen_stats_N${N}_SIMMEM${SIMMEM}_FROMMEM${FROMMEM}.pdf

#~ python3 /home/gonthier/these_gonthier_maxime/Code/Barplot_DARTS.py Output_maxime/DARTS_data_choosen_stats_frommem.csv
#~ mv Output_maxime/DARTS_data_choosen_stats_frommem.csv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/Matrice3D/DARTS_data_choosen_stats_frommem.csv
#~ mv plot.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/Matrice3D/DARTS_data_choosen_stats_frommem.pdf

#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER))
#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER))
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ /./home/gonthier/these_gonthier_maxime/Code/permutation_visu_python $((N)) ${ORDO} 1 1
#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 4

#~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."

# Pour tester sur Grid5k	
# ulimit -S -s 5000000
# source ../.bashrc
# ORDO=dynamic-data-aware ; NITER=1 ; N=20 ; NGPU=1 ; CM=500 ; CP=5 ; TH=10 ; i=1 ; TRACE=1 ; HOST=gemini-1-fgcs ; STARPU_SCHED=${ORDO} STARPU_GENERATE_TRACE=$((TRACE)) PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
# ORDO=modular-eager-prefetching ; NITER=1 ; N=20 ; NGPU=1 ; CM=500 ; CP=5 ; TH=10 ; i=1 ; TRACE=1 ; HOST=gemini-1-fgcs ; STARPU_SCHED=${ORDO} STARPU_GENERATE_TRACE=$((TRACE)) STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))



