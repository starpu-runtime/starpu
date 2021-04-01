#!/usr/bin/bash

#Script supposed to launch anything as long as I evaluate perfomances in GFlops on the Y-axis with the right arguments.
#This way I have only one script and one R file in /home/gonthier/these_gonthier_maxime/Starpu/R/ScriptR/

#To use it: 
#~/starpu$ bash Scripts_maxime/performances.sh starpu_path scriptR_path application x-axis nb_of_gpu number_of_size_tested scale algos_to_test name_of_file_to_save raw_file

PATH_STARPU=$1 #/home/user/
PATH_R=$2 #/home/user/these_gonthier_maxime/Starpu/R/ScriptR/
APPLICATION=$3 #Matrice_ligne / Matrice3D / Cholesky / Random_tasks
X_AXIS=$4 #Working_set / Threshold / Memory
NB_GPU=$5 #1 / 3
NB_TAILLE_TESTE=$6 #From 1 to 10 usually
ECHELLE=$7 #How much to we grow N at each iteration
ALGOS=$8
FICHIER=$9 #A name with abreviation explained in Readme.md
FICHIER_RAW=$10 #Use two different raw file if you run two execution in parallel in order to not to overwrite a file we are using

export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000

if [ $APPLICATION = "Matrice_ligne" ]
	then
	if [ $X_AXIS = "Threshold" ]
		then
		if [ $GPU = "1" ]
			then
		fi
		if [ $GPU = "3" ]
			then
			echo "############## HFP ##############"
			for ((i=0 ; i<(($NB_TAILLE_TESTE)); i++)) do
				N=$((i*ECHELLE_X))
				STARPU_SCHED=HFP READY=1 MULTIGPU=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NTASKS_THRESHOLD=N STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
			"############## HFP + LOAD BALANCE ##############"
			for ((i=0 ; i<(($NB_TAILLE_TESTE)); i++)) do
				N=$((i*ECHELLE_X))
				STARPU_SCHED=HFP READY=1 MULTIGPU=4 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NTASKS_THRESHOLD=N STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
			"############## HFP + LOAD BALANCE + HFP ##############"
			for ((i=0 ; i<(($NB_TAILLE_TESTE)); i++)) do
				N=$((i*ECHELLE_X))
				STARPU_SCHED=HFP READY=1 MULTIGPU=5 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NTASKS_THRESHOLD=N STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
		fi
	fi
fi

# Tracage des GFlops
gcc -o get_difference_between_orders get_difference_between_orders.c
./get_difference_between_orders Output_maxime/Task_order_HFP_0 Output_maxime/Task_order_effective_0 ${PATH_R}/R/Data/${DOSSIER}/Difference_between_orders/${FICHIER:0}.txt
Rscript ${PATH_R}/R/ScriptR/Difference_between_orders/Diff_HFP_HEFT_BW350_CM500.R ${PATH_R}/R/Data/${DOSSIER}/Difference_between_orders/${FICHIER}.txt $((GPU)) $((NT))
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Difference_between_orders/${FICHIER:0}.pdf
