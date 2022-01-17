NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
NB_ALGO_TESTE=$5
NAME=$6
PATH_STARPU=$7
PATH_R=$8
START_X=0
GPU=gemini-1-fgcs
NITER=11

if [ $DOSSIER == "Matrice_ligne" ] && [ $MODEL == "HFP" ]
	then
	echo "Tracage de HFP M2D"
	ECHELLE_X=$((5*NGPU))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_M2D.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M2D.txt
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_HFP_M2D.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M2D.txt
	
	# Tracage des GFlops 1
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M2D.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
	
	# Tracage des GFlops 2
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_WITH_SCHEDULING_TIME_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_WITH_SCHEDULING_TIME_FGCS_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M2D.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

if [ $MODEL == "HFP_memory" ]
	then
	echo "Tracage de HFP MEMORY"
	ECHELLE_X=$((50))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_MEMORY_M2D.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_MEMORY_M2D.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_MEMORY_M2D.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

if [ $DOSSIER == "Matrice3D" ]
	then
	echo "Tracage de HFP M3D"
	ECHELLE_X=$((5*NGPU))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_M3D.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M3D.txt
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_HFP_M3D.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M3D.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M3D.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M3D.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

if [ $DOSSIER == "Cholesky" ]
	then
	echo "Tracage de HFP CHO"
	ECHELLE_X=$((5*NGPU))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_CHO.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_CHO.txt
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_HFP_CHO.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_CHO.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GFlops_raw_out_3.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

# HFP M2D RANDOM TASK ORDER
if [ $DOSSIER == "Random_task_order" ]
	then
	echo "Tracage de HFP RANDOM TASK ORDER"
	ECHELLE_X=$((5*NGPU))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_M2D_RANDOM_ORDER.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M2D_RANDOM_ORDER.txt
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_HFP_M2D_RANDOM_ORDER.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M2D_RANDOM_ORDER.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M2D_RANDOM_ORDER.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M2D_RANDOM_ORDER.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

# HFP M2D RANDOM TASKS
if [ $DOSSIER == "Random_tasks" ]
	then
	echo "Tracage de HFP RANDOM TASKS"
	ECHELLE_X=$((5*NGPU))
		
	#~ scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_M2D_RANDOM_TASKS.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M2D_RANDOM_TASKS.txt
	#~ scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_HFP_M2D_RANDOM_TASKS.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M2D_RANDOM_TASKS.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_M2D_RANDOM_TASKS.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_M2D_RANDOM_TASKS.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

# HFP M2D SPARSE
if [ $DOSSIER == "Sparse" ]
	then
	echo "Tracage de HFP SPARSE"
	ECHELLE_X=$((50*NGPU))
		
	#~ scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_SPARSE.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_SPARSE.txt
	#~ scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_HFP_SPARSE.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_SPARSE.txt
	
	# Tracage des GFlops 1 
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_SPARSE.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
	
	# Tracage des GFlops 2
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_SPARSE.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_WITH_SCHEDULING_TIME_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_WITH_SCHEDULING_TIME_FGCS_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_SPARSE.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

# HFP M2D SPARSE INFINIE
if [ $DOSSIER == "Sparse_mem_infinite" ]
	then
	echo "Tracage de HFP SPARSE INFINITE"
	ECHELLE_X=$((50*NGPU))
		
	#~ scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_HFP_SPARSE_INFINIE.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_SPARSE_INFINIE.txt
	#~ scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_HFP_SPARSE_INFINIE.txt ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_SPARSE_INFINIE.txt
	
	# Tracage des GFlops 1
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_SPARSE_INFINIE.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage des GFlops 2
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/GF_HFP_SPARSE_INFINIE.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_WITH_SCHEDULING_TIME_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_WITH_SCHEDULING_TIME_FGCS_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/FGCS/DT_HFP_SPARSE_INFINIE.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
