#~ Pour récup les data sur Grid5k et les process pour IPDPS

#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_FGCS.sh 13 Matrice_ligne HFP 1 9
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_FGCS.sh 10 Matrice_ligne HFP_memory 1 9
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_FGCS.sh 8 Matrice3D HFP 1 9
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_FGCS.sh 8 Cholesky HFP 1 9

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
NB_ALGO_TESTE=$5
START_X=0
GPU=gemini-1-fgcs
PATH_R=/home/gonthier/these_gonthier_maxime/Starpu
PATH_STARPU=/home/gonthier
NITER=11

if [ $MODEL == "HFP" ]
	then
	ECHELLE_X=$((5*NGPU))
		
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_3.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_4.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_4.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/HFP_time.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/HFP_time.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_3.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage du temps
	gcc -o cut_time_raw_out cut_time_raw_out.c
	./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_4.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt TIME_${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
	
	# Tracage du temps de HFP
	mv ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/HFP_time.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/HFP_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/HFP_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt HFP_TIME_HFP ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/HFP_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
if [ $MODEL == "HFP_memory" ]
	then
	ECHELLE_X=$((50))
		
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt
	
	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_FGCS ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
