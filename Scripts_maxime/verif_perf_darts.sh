# bash Scripts_Maxime/verif_perf_darts.sh

start=`date +%s`

#define THRESHOLD /* Pour arrêter de regarder dans la liste des données plus tôt. 0 = no threshold, 1 = threshold à 14400 tâches pour une matrice 2D (donc APP == 0) et à 1599 tâches aussi pour matrice 3D (donc APP == 1), 2 = on s'arrete des que on a trouvé 1 donnée qui permet de faire au moins une tache gratuite ou si il y en a pas 1 donnée qui permet de faire au moins 1 tache a 1 d'ere gratuite. 0 par défaut */
#define APP /* 0 matrice 2D. 1 matrice 3D. Sur 1 on regarde les tâches à 1 d'être gratuite galement. Pas plus loin. */
#define CHOOSE_BEST_DATA_FROM /* Pour savoir où on regarde pour choisir la meilleure donnée. 0, on regarde la liste des données pas encore utilisées. 1 on regarde les données en mémoire et à partir des tâches de ces données on cherche une donnée pas encore en mémoire qui permet de faire le plus de tâches gratuite ou 1 from free. */
#define SIMULATE_MEMORY /* Default 0, means we use starpu_data_is_on_node, 1 we also look at nb of task in planned and pulled task. */
#define TASK_ORDER /* 0, signifie qu'on randomize entièrement la liste des tâches. 1 je ne randomise que les nouvelles tâches entre elle et les met à la fin des listes de taches. 2 je ne randomise pas et met chaque GPU sur un m/NGPU portion différentes pour qu'ils commencent à différent endroit de la liste de tâches. */
#define DATA_ORDER /* 0, signifie qu'on randomize entièrement la liste des données. 1 je ne randomise que les nouvelles données entre elle et les met à la fin des listes de données. 2 je ne randomise pas et met chaque GPU sur un Ndata/NGPU portion différentes pour qu'ils commencent à différent endroit de la liste de données.*/
#define DEPENDANCES /* 0 non, 1 utile pour savoir si on fais des points de départs différents dans main task list (on ne le fais pas si il y a des dependances) */
#define PRIO /* 0 non, 1 tiebreak data selection with the that have the highest priority task */

make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

NITER=11
CM=500
TH=10
CP=5
HOST="gemini-1-fgcs-36"

echo "Matrice 2D - 1GPU - N40: 12895.3"

STARPU_HOSTNAME=${HOST} STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/mult/sgemm --no-prio -xy $((960*40)) -nblocks 40 -iter 1

#~ echo "Matrice 2D - 2GPU - N40: 25698.8"

#~ STARPU_HOSTNAME=${HOST} STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=2 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1

#~ echo "Cholesky - 1GPU - N40 - APP0: 3917.0"

#~ STARPU_HOSTNAME=${HOST} STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*40)) -nblocks 40

#~ echo "Cholesky - 1GPU - N40 - APP1 - DEP1 - PRIO1: 4001.3"

#~ DEPENDANCES=1 PRIO=1 APP=1 STARPU_HOSTNAME=${HOST} STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*40)) -nblocks 40

#~ echo "Matrice 3D - 1GPU - N40 - APP1: 11936.2"

#~ APP=1 STARPU_HOSTNAME=${HOST} STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*40)) -nblocks 40 -z $((960*4)) -nblocksz $((4)) -iter 1

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
