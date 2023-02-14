# oarsub -t exotic -p "network_address in ('gemini-1.lyon.grid5000.fr')" -r '2023-02-14 18:00:00' -l walltime=01:00:00 "bash Scripts_maxime/PlaFRIM-Grid5k/all_complexity.sh"

# bash Scripts_maxime/PlaFRIM-Grid5k/darts_complexity.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL

# bash Scripts_maxime/PlaFRIM-Grid5k/darts_complexity.sh 1 1920 12 2000 best_ones
# bash Scripts_maxime/PlaFRIM-Grid5k/darts_complexity.sh 2 1920 12 2000 best_ones
# bash Scripts_maxime/PlaFRIM-Grid5k/darts_complexity.sh 4 1920 12 2000 best_ones
# bash Scripts_maxime/PlaFRIM-Grid5k/darts_complexity.sh 8 1920 12 2000 best_ones

if [ $# != 5 ]
then
	echo "Arguments must be: bash Scripts_maxime/darts_complexity.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL"
	exit
fi

make -j 6
START_X=0
ECHELLE_X=5
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
ulimit -S -s 50000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_RAW_DT}
truncate -s 0 ${FICHIER_BUS}

NGPU=$1

TH=10
CP=5

SEED=1

NITER=1

TAILLE_TUILE=$2

CM=$4
MODEL=$5

echo "CM =" ${CM} "BLOCK SIZE =" ${TAILLE_TUILE} "NGPU =" ${NGPU}

NCOMBINAISONS=4
if [ NGPU != 1 ]
then
	NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))
fi

# Best results is obtained with following parameters so far:
# EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1
# THRESHOLD=0
# APP=1
# CHOOSE_BEST_DATA_FROM=0 (need to test with 1 in real)
# SIMULATE_MEMORY=0 (need to test with 1 in real)
# TASK_ORDER=1-2
# DATA_ORDER=1-2
# STARPU_SCHED_READY=1
# DEPENDANCES=1
# PRIO=1
# FREE_PUSHED_TASK_POSITION=1
# DOPT_SELECTION_ORDER=1
# PRIORITY_ATTRIBUTION=1 (1 is bottom level)
# GRAPH_DESCENDANTS=0
# HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=0or1 idk

# Best ones
algo1="DARTS"
algo2="DARTS highest prio"

echo "N,${algo1},${algo2}" > Output_maxime/Legende.txt

NB_TAILLE_TESTE=$3

NB_ALGO_TESTE=2
PRIORITY_ATTRIBUTION=1

# Best ones
echo "#### ${algo1} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	echo "N=${N}"
	PRIORITY_ATTRIBUTION=$((PRIORITY_ATTRIBUTION)) CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 STARPU_LIMIT_CUDA_MEM=$((CM)) HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=0 GRAPH_DESCENDANTS=0 DOPT_SELECTION_ORDER=1 STARPU_SCHED_READY=1 PRIORITY_ATTRIBUTION=0 TASK_ORDER=2 DATA_ORDER=2 FREE_PUSHED_TASK_POSITION=1 DEPENDANCES=1 PRIO=1 APP=1 SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((TAILLE_TUILE*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
done
echo "#### ${algo2} ####"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	echo "N=${N}"
	PRIORITY_ATTRIBUTION=$((PRIORITY_ATTRIBUTION)) CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 STARPU_LIMIT_CUDA_MEM=$((CM)) HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=1 GRAPH_DESCENDANTS=0 DOPT_SELECTION_ORDER=1 STARPU_SCHED_READY=1 PRIORITY_ATTRIBUTION=0 TASK_ORDER=2 DATA_ORDER=2 FREE_PUSHED_TASK_POSITION=1 DEPENDANCES=1 PRIO=1 APP=1 SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((TAILLE_TUILE*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
done

echo "Converting data"
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/Cholesky_dependances/GF_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${CM}Mo.csv

mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/Cholesky_dependances/DT_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${CM}Mo.csv
