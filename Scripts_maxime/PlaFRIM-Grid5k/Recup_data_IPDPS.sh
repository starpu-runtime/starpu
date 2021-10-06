#~ Pour récup les data sur Grid5k et les process pour IPDPS
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp 1 6
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp 2 7

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
NB_ALGO_TESTE=$5
START_X=0
ECHELLE_X=$((5*NGPU))
GPU=gemini-2-ipdps
PATH_R=/home/gonthier/these_gonthier_maxime/Starpu
PATH_STARPU=/home/gonthier

#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_1.txt
#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_3.txt
#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_4.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_4.txt
#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/Schedule_time_raw_out.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/Schedule_time_raw_out.txt
#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/DDA_eviction_time.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/DDA_eviction_time.txt

#Tracage data transfers
gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_3.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ # Tracage du temps
gcc -o cut_time_raw_out cut_time_raw_out.c
./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_4.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

# Tracage du temps de schedule
./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/Schedule_time_raw_out.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt SCHEDULE_TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

# Tracage du temps d'éviction et de schedule de DDA
Rscript ${PATH_R}/R/ScriptR/GF_X.R /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/DDA_eviction_time.txt EVICTION_TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/EVICTION_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
mv /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/DDA_eviction_time.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/EVICTION_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
