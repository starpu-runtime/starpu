# bash Scripts_maxime/quick_plot1.sh
# bash Scripts_maxime/quick_plot2.sh

make -j 6
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

START_X=0
ECHELLE_X=5
FICHIER_RAW=/home/gonthier/starpu/Output_maxime/GFlops_raw_out_1.txt
ulimit -S -s 50000000
truncate -s 0 ${FICHIER_RAW}

NGPU=1

CM=500
#~ CM=0 # 0 = infinie
#~ CM=100

EVICTION=0
#~ EVICTION=1

READY=0
#~ READY=1

TH=10

CP=5

HOST="gemini-1-fgcs-36"

SEED=1

NITER=1

TAILLE_TUILE=960

#~ APP3D=0
APP3D=1

SPARSE=0
#~ SPARSE=10

TASK_ORDER=0
#~ TASK_ORDER=1
#~ TASK_ORDER=2

DATA_ORDER=0
#~ DATA_ORDER=1
#~ DATA_ORDER=2

algo1="DARTS + LUF + 3D + PRIO -no-prio"
algo2="DMDAR"

echo "N,${algo1},${algo2}" > Output_maxime/Legende.txt

NB_TAILLE_TESTE=10
NB_ALGO_TESTE=2

echo "#### ${algo1} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	DEPENDANCES=1 PRIO=1 APP=$((APP3D)) SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED_READY=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) -no-prio | tail -n 1 >> ${FICHIER_RAW}
	#~ DEPENDANCES=1 PRIO=1 APP=$((APP3D)) SEED=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED_READY=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) -no-prio
done

echo "#### ${algo2} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} SEED=$((N/5)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	#~ STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
done

echo "Converting data"
gcc -o cut_gflops_raw_out_csv cut_gflops_raw_out_csv.c
./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot.csv
