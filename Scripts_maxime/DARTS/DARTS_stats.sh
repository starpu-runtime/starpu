# bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 12 Cholesky_dependances Stats gemini-1-cho_dep_corrected 4

# bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 12 Cholesky_dependances Data_choosen gemini-1-cho_dep_corrected 4

# Il faut que sois définie #define PRINT_STATS dans le .h

make -j 100
PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
GPU=$6
NGPU=$7
START_X=0
ECHELLE_X=5
HOST=$GPU
CM=2000
TH=10
CP=5

start=`date +%s`
make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

echo "N,Nb conflits,Nb conflits critiques" > Output_maxime/Data/DARTS/Nb_conflit_donnee.csv	
echo "N,Return NULL, Return task, Return NULL because main task list is empty,Nb of random selection,Nb 1 from free task not found after choosing a data,Nb of free data found,Nb of 1 from free data found" > Output_maxime/Data/DARTS/Choice_during_scheduling.csv
echo "N,victim_selector_refused_not_on_node,victim_selector_refused_cant_evict,victim_selector_return_refused,victim_selector_return_unvalid,victim_selector_return_data_not_in_planned_and_pulled,victim_evicted_compteur,victim_selector_compteur,victim_selector_return_no_victim,victim_selector_belady" > Output_maxime/Data/DARTS/Choice_victim_selector.csv
echo "N,Nb refused tasks,Nb times a new set of task is initialized" > Output_maxime/Data/DARTS/Misc.csv
echo "N,Choose eviction,Eviction failed,Belady,Scheduling,Choosing the best data,Filling planned task,Initialisation,Randomize,Choix tâche aléatoire,Least_used_data_planned_task,Temps total" > Output_maxime/Data/DARTS/DARTS_time.csv

ORDO="dynamic-data-aware"

EVICTION=1

READY=0

APP3D=0
#~ APP3D=1

TASK_ORDER=0
#~ TASK_ORDER=1
#~ TASK_ORDER=2

DATA_ORDER=0
#~ DATA_ORDER=1
#~ DATA_ORDER=2

THRESHOLD=0
#~ THRESHOLD=2

TAILLE_TUILE=1920

FM=0

HOST="gemini-1-cho_dep_corrected"

SM=0

if [ $MODEL = "Stats" ]
	then

	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do
		N=$((START_X+i*ECHELLE_X))
		echo "####################\o/\o/\o/\o/ N =" $((N)) "\o/\o/\o/\o/####################"
		PRINT_N=$((N)) PRINT_TIME=2 PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK=2 CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 STARPU_LIMIT_CUDA_MEM=$((CM)) HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE=1 GRAPH_DESCENDANTS=0 DOPT_SELECTION_ORDER=1 STARPU_SCHED_READY=1 PRIORITY_ATTRIBUTION=$((1)) TASK_ORDER=2 DATA_ORDER=2 FREE_PUSHED_TASK_POSITION=1 DEPENDANCES=1 PRIO=1 APP=1 SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((TAILLE_TUILE*N)) -nblocks $((N))
	done

	# Traçage avec python
	echo "##### Drawing... #####"
	python3 /home/gonthier/these_gonthier_maxime/Code/BarPlot.py Output_maxime/Data/DARTS/Nb_conflit_donnee.csv 1
	mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Stats/Conflits_${GPU}_${NGPU}GPU.pdf
	python3 /home/gonthier/these_gonthier_maxime/Code/BarPlot.py Output_maxime/Data/DARTS/Choice_during_scheduling.csv 1
	mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Stats/Choix_scheduling_${GPU}_${NGPU}GPU.pdf
	python3 /home/gonthier/these_gonthier_maxime/Code/BarPlot.py  Output_maxime/Data/DARTS/Choice_victim_selector.csv 1
	mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Stats/Choix_eviction_${GPU}_${NGPU}GPU.pdf
	python3 /home/gonthier/these_gonthier_maxime/Code/BarPlot.py Output_maxime/Data/DARTS/Misc.csv 1
	mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Stats/Misc_${GPU}_${NGPU}GPU.pdf
	python3 /home/gonthier/these_gonthier_maxime/Code/Simple_Plot.py Output_maxime/Data/DARTS/DARTS_time.csv 1
	mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Stats/Temps_${GPU}_${NGPU}GPU.pdf
fi
if [ $MODEL = "Data_choosen" ]
	then

	N=30
	echo "####################\o/\o/\o/\o/ N =" $((N)) "\o/\o/\o/\o/####################"
	TASK_ORDER=$((TASK_ORDER)) DATA_ORDER=$((DATA_ORDER)) PRINT_N=$((N)) DEPENDANCES=1 THRESHOLD=$((THRESHOLD)) CHOOSE_BEST_DATA_FROM=$((FM)) SIMULATE_MEMORY=$((SM)) APP=$((APP3D)) STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=$((READY)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((1920*N)) -nblocks $((N))

	# Traçage avec python
	echo "##### Drawing... ####"
	python3 /home/gonthier/these_gonthier_maxime/Code/Barplot_datachoosen.py Output_maxime/Data/DARTS/DARTS_data_choosen_stats_GPU_1.csv
	mv ${PATH_STARPU}/starpu/plot1.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Stats/Data_choosen_${GPU}_${NGPU}GPU.pdf
	mv ${PATH_STARPU}/starpu/plot2.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Stats/Nb_task_added_in_planned_task_${GPU}_${NGPU}GPU.pdf
fi

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
