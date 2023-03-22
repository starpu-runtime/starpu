# bash Scripts_maxime/PlaFRIM-Grid5k/out_of_core_lu.sh 4096 10 2 1000 modular-eager-prefetching
# bash Scripts_maxime/PlaFRIM-Grid5k/out_of_core_lu.sh 4096 10 2 1000 dmdas
# bash Scripts_maxime/PlaFRIM-Grid5k/out_of_core_lu.sh 4096 10 2 1000 lws
# bash Scripts_maxime/PlaFRIM-Grid5k/out_of_core_lu.sh 4096 10 2 1000 dynamic-data-aware

module load openmpi/4.1.3_gcc-10.2.0
bash script_initialisation_sans_simgrid_grid5k

#~ bus_sttas marche ? A tester
#~ lancer ca en bash de oarsub
# PENSER A CALIBRATE AVEC DMDAR!!

if [ $# != 5 ]
then
	echo "Arguments must be: TAILLE_TUILE NB_TAILLE_TESTE ECHELLE_X MEMOIRE SCHEDULER"
	exit
fi

START_X=0
ECHELLE_X=$3
FICHIER_RAW=Output_maxime/out_of_core_raw.txt
truncate -s 0 ${FICHIER_RAW}

TAILLE_TUILE=$1
NB_TAILLE_TESTE=$2
CM=$4
SCHEDULER=$5
TH=10
CP=5
SEED=1

echo ${CM}"Mo -" ${TAILLE_TUILE} "tiles -" ${NB_TAILLE_TESTE} "tailles testées -" ${SCHEDULER}
	
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	echo "N=${N}"
	if [ ${SCHEDULER} == "dynamic-data-aware" ]; then
		CPU_ONLY=1 THRESHOLD=0 DOPT_SELECTION_ORDER=7 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET=0 PRIORITY_ATTRIBUTION=1 HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=1 GRAPH_DESCENDANTS=0 PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=2 STARPU_SCHED_READY=1 TASK_ORDER=2 DATA_ORDER=2 FREE_PUSHED_TASK_POSITION=1 DEPENDANCES=1 PRIO=1 APP=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_DISK_SWAP_BACKEND=unistd_o_direct STARPU_LIMIT_CPU_MEM=$((CM)) STARPU_DISK_SWAP=/tmp STARPU_NCUDA=0 SEED=$((N/5)) STARPU_SCHED=${SCHEDULER} STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NOPENCL=0 ./mpi/examples/mpi_lu/plu_outofcore_example_double -size $((TAILLE_TUILE*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	else
		STARPU_DISK_SWAP_BACKEND=unistd_o_direct STARPU_LIMIT_CPU_MEM=$((CM)) STARPU_DISK_SWAP=/tmp STARPU_NCUDA=0 SEED=$((N/5)) STARPU_SCHED=${SCHEDULER} STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NOPENCL=0 ./mpi/examples/mpi_lu/plu_outofcore_example_double -size $((TAILLE_TUILE*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	fi
done

mv Output_maxime/out_of_core_raw.txt Output_maxime/Data/Out_of_core_lu/GF_${TAILLE_TUILE}_${CM}Mo_${SCHEDULER}.csv
