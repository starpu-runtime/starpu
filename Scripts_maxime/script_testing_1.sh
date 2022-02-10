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

#~ make -C src/ -j 6
make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

N=5
#~ N=15
N=40

NGPU=1
NGPU=2
NGPU=3
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
#~ CM=200

EVICTION=0
EVICTION=1

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
NITER=2
NITER=8
NITER=11

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

#~ truncate -s 0 "Output_maxime/DARTS_data_choosen_stats.txt"
#~ truncate -s 0 "Output_maxime/DARTS_data_choosen_stats_simmem.txt"
#~ truncate -s 0 "Output_maxime/DARTS_data_choosen_stats_frommem.txt"
#~ truncate -s 0 "Output_maxime/DARTS_data_choosen_stats_frommem_simmem.txt"
#~ A CORRIGER pour from mem on lis pas autant!!
#~ N=5
CHOOSE_BEST_DATA_FROM=0 MULTIGPU=$((MULTI)) REVERSE=1 FASTER_FIRST_ITERATION=0 RANDOM_TASK_ORDER=0 STARPU_SCHED_READY=$((READY)) BELADY=$((BELADY)) ORDER_U=1 SIMULATE_MEMORY=0 PRINT3D=0 PRINT_N=$((N)) STARPU_GENERATE_TRACE=0 APP=$((APP3D)) STARPU_SCHED=${ORDO} STARPU_HOSTNAME=${HOST} EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER))

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
# ORDO=dynamic-data-aware ; NITER=8 ; N=40 ; NGPU=4 ; CM=500 ; CP=5 ; TH=10 ; i=1 ; EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=${ORDO} STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER))
# ORDO=dynamic-data-aware ; NITER=8 ; N=100 ; NGPU=2 ; CM=500 ; CP=5 ; TH=10 ; i=1 ; EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=${ORDO} STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 libtool --mode=execute gdb --args ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER))
# ORDO=dynamic-data-aware ; NITER=8 ; N=100 ; NGPU=2 ; CM=500 ; CP=5 ; TH=10 ; i=1 ; STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=0 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 NATURAL_ORDER=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER))
