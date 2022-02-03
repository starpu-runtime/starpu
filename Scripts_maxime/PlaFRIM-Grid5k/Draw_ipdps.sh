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

if [ $DOSSIER == "Matrice_ligne" ]
	then
	echo "Tracage de M2D"
	ECHELLE_X=$((5*NGPU))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_DARTS_M2D_${NGPU}GPU.txt ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_M2D_${NGPU}GPU.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_M2D_${NGPU}GPU.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
	
	if [ $NGPU != 4 ]
	then
		scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/DT_DARTS_M2D_${NGPU}GPU.txt ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/DT_HFP_M2D.txt
		
		# Tracage data transfers
		gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
		./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/DT_HFP_M2D.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
		Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU} ${NITER}
		mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
	fi
fi

if [ $DOSSIER == "Cholesky" ]
	then
	echo "Tracage de HFP CHO"
	ECHELLE_X=5
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_DARTS_CHO_4GPU.txt ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_CHO_4GPU.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_CHO_4GPU.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

if [ $DOSSIER == "Random_task_order" ]
	then
	echo "Tracage de RANDOM TASK ORDER"
	ECHELLE_X=5
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_DARTS_M2D_RANDOM_ORDER_2GPU.txt ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_M2D_RANDOM_ORDER_2GPU.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_M2D_RANDOM_ORDER_2GPU.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

if [ $DOSSIER == "Sparse" ]
	then
	echo "Tracage de SPARSE"
	ECHELLE_X=50
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_DARTS_SPARSE_4GPU.txt ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_SPARSE_4GPU.txt
	
	# Tracage des GFlops 1 
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_SPARSE_4GPU.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi

if [ $DOSSIER == "Sparse_mem_infinite" ]
	then
	echo "Tracage de SPARSE INFINITE"
	ECHELLE_X=50
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/Data/GF_DARTS_SPARSE_INFINIE_4GPU.txt ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_SPARSE_INFINIE_4GPU.txt
	
	# Tracage des GFlops 1
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/IPDPS/GF_DARTS_SPARSE_INFINIE_4GPU.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
