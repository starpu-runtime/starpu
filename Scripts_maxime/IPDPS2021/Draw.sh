NAME=$1
NGPU=$2
PATH_STARPU=$3
PATH_R=$4
START_X=0
GPU=gemini-2-ipdps
PATH_R=/home/gonthier/these_gonthier_maxime/Starpu
PATH_STARPU=/home/gonthier
NITER=11
MODEL=dynamic_data_aware_no_hfp
DOSSIER=Matrice_ligne

if [ $NGPU = 1 ]
then
	NB_TAILLE_TESTE=16
	NB_ALGO_TESTE=5
fi
if [ $NGPU = 1 ]
then
	NB_TAILLE_TESTE=15
	NB_ALGO_TESTE=6
fi
if [ $NGPU = 1 ]
then
	NB_TAILLE_TESTE=10
	NB_ALGO_TESTE=6
fi

ECHELLE_X=$((5*NGPU))
	
scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt ${PATH_STARPU}/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_1.txt
scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt ${PATH_STARPU}/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_3.txt
scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops_raw_out_4.txt ${PATH_STARPU}/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_4.txt

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_1.txt ${PATH_R}/locality-aware-scheduling/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/locality-aware-scheduling/R/ScriptR/GF_X.R ${PATH_R}/locality-aware-scheduling/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU} ${NITER}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/locality-aware-scheduling/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

#Tracage data transfers
gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_3.txt ${PATH_R}/locality-aware-scheduling/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/locality-aware-scheduling/R/ScriptR/GF_X.R ${PATH_R}/locality-aware-scheduling/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/locality-aware-scheduling/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

# Tracage du temps
gcc -o cut_time_raw_out cut_time_raw_out.c
./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_4.txt ${PATH_R}/locality-aware-scheduling/R/Data/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/locality-aware-scheduling/R/ScriptR/GF_X.R ${PATH_R}/locality-aware-scheduling/R/Data/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt TIME_${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU} ${NITER}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/locality-aware-scheduling/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
