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

if [ $MODEL == "HFP" ]
	then
	ECHELLE_X=$((5*NGPU))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_3.txt
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops_raw_out_4.txt ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_4.txt
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/HFP_time.txt ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/HFP_time.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_3.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
if [ $MODEL == "HFP_memory" ]
	then
	ECHELLE_X=$((50))
		
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
