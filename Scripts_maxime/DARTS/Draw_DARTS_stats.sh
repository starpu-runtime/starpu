#!/bin/bash
#	bash Scripts_maxime/DARTS/Draw_DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Cholesky_dependances DARTS gemini-1-fgcs 1

PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
GPU=$6
NGPU=$7

#~ # Tracage des GFlops
#~ gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
#~ ./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ if [[ $MODEL != "dynamic_data_aware_no_hfp_no_mem_limit" ]]
#~ then
	#Tracage data transfers
	#~ gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	#~ ./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT} ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage du temps
	#~ gcc -o cut_time_raw_out cut_time_raw_out.c
	#~ ./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_TIME} ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage du temps global et du temps de schedule all et split
	#~ gcc -o cut_schedule_time_raw_out cut_schedule_time_raw_out.c
	#~ ./cut_schedule_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_SCHEDULE_TIME} ${PATH_R}/R/Data/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt SCHEDULE_TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

	#~ if [[ $MODEL != "dynamic_data_aware_no_hfp_sparse_matrix" ]]
	#~ then
		#~ # A retirer surement : tracage du temps d'éviction de DDA et de schedule de DDA
		#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/DARTS_time.txt DARTS_time_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
		#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DARTS_time_${MODEL}_${GPU}_${NGPU}GPU.pdf
		#~ mv ${PATH_STARPU}/starpu/Output_maxime/DARTS_time.txt ${PATH_R}/R/Data/${DOSSIER}/DARTS_time_${MODEL}_${GPU}_${NGPU}GPU.txt
		#~ # No threshold
		#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold.txt DARTS_time_no_threshold_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
		#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DARTS_time_no_threshold_${MODEL}_${GPU}_${NGPU}GPU.pdf
		#~ mv ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold.txt ${PATH_R}/R/Data/${DOSSIER}/DARTS_time_no_threshold_${MODEL}_${GPU}_${NGPU}GPU.txt
		#~ # No threshold and choose best data from memory
		#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold_choose_best_data_from_memory.txt DARTS_time_no_threshold_choose_best_data_from_memory_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
		#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DARTS_time_no_threshold_choose_best_data_from_memory_${MODEL}_${GPU}_${NGPU}GPU.pdf
		#~ mv ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold_choose_best_data_from_memory.txt ${PATH_R}/R/Data/${DOSSIER}/DARTS_time_no_threshold_choose_best_data_from_memory_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ fi
#~ fi

# Plot python GF sans légende
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv 0
mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU_sanslegende.pdf
# Plot python DT sans légende
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.csv 0
mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU_sanslegende.pdf

# Plot python GF
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv 1
mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
# Plot python DT
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.csv 1
mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

