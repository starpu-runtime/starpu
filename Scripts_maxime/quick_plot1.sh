# bash Scripts_maxime/quick_plot1.sh 1
# bash Scripts_maxime/quick_plot2.sh 1

make -j 6
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

START_X=0
ECHELLE_X=5
FICHIER_RAW=/home/gonthier/starpu/Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=/home/gonthier/starpu/Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=/home/gonthier/starpu/Output_maxime/GFlops_raw_out_3.txt
ulimit -S -s 50000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_RAW_DT}
truncate -s 0 ${FICHIER_BUS}

NGPU=$1

CM=500
#~ CM=0 # 0 = infinie
#~ CM=100

TH=10

CP=5

HOST="gemini-1-fgcs-36"

SEED=1

NITER=1

TAILLE_TUILE=960

#~ APP3D=0
APP3D=1

NCOMBINAISONS=4
if [ NGPU != 1 ]
then
	NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))
fi

algo1="DARTS+LUF+3D -no-prio"
algo2="DARTS+LUF+3D+NTO -no-prio"
algo3="DARTS+LUF+3D+NTO+NDO -no-prio"
algo4="DMDAR"

echo "N,${algo1},${algo2},${algo3},${algo4}" > Output_maxime/Legende.txt

NB_TAILLE_TESTE=9
NB_ALGO_TESTE=4

echo "#### ${algo1} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	DEPENDANCES=1 PRIO=1 APP=$((APP3D)) SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED_READY=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) -no-prio | tail -n 1 >> ${FICHIER_RAW}
	sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
done

echo "#### ${algo2} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	TASK_ORDER=2 DATA_ORDER=0 DEPENDANCES=1 PRIO=1 APP=$((APP3D)) SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED_READY=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) -no-prio | tail -n 1 >> ${FICHIER_RAW}
	sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
done

echo "#### ${algo3} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	TASK_ORDER=2 DATA_ORDER=2 DEPENDANCES=1 PRIO=1 APP=$((APP3D)) SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED_READY=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) -no-prio | tail -n 1 >> ${FICHIER_RAW}
	sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
done

echo "#### ${algo4} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} SEED=$((N/5)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
done

echo "Converting data"
gcc -o cut_gflops_raw_out_csv cut_gflops_raw_out_csv.c
./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot.csv
gcc -o cut_datatransfers_raw_out_csv cut_datatransfers_raw_out_csv.c
./cut_datatransfers_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT} /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot_dt.csv
