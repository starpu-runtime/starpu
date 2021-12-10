#~ Pour récup les data sur Grid5k et les process pour IPDPS

#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp 1 6 V
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp 2 5 V
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp 3 5 V
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp 4 5 V

# Random
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 13 Random_task_order dynamic_data_aware_no_hfp 2 5 V

#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_profiling 1 5 X
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_profiling 2 6 X
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 10 Matrice_ligne dynamic_data_aware_no_hfp_profiling 3 6 X

#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 11 Matrice_ligne dynamic_data_aware_compare_threshold 2 1
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 11 Matrice_ligne dynamic_data_aware_compare_threshold_worse_time 2 1
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 12 Matrice_ligne dynamic_data_aware_compare_threshold_type 2 5
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 10 Matrice_ligne dynamic_data_aware_compare_choose_best_data_type 2 3

#~ AVEC 1 je fais le global time et eviction time et les GF avec l'autre les DT et le SCHEDULE

#Pour le rebuttal
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 3 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1 3
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 10 Matrice3D dynamic_data_aware_no_hfp 1 8
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 8 Matrice3D dynamic_data_aware_no_hfp 2 9
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 10 Cholesky dynamic_data_aware_no_hfp 1 8
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_IPDPS.sh 7 Cholesky dynamic_data_aware_no_hfp 2 9

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
NB_ALGO_TESTE=$5
START_X=0
#~ GPU=gemini-2-ipdps
GPU=gemini-1-fgcs
PATH_R=/home/gonthier/these_gonthier_maxime/Starpu
PATH_STARPU=/home/gonthier
NITER=11

if [ $MODEL == "dynamic_data_aware_no_hfp_no_mem_limit" ]
	then
	GPU=gemini-1-fgcs
	if [ $DOSSIER == "Matrice_ligne" ]
	then
		ECHELLE_X=$((15*NGPU))
	fi
	
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt

	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
if [ $MODEL == "dynamic_data_aware_no_hfp" ]
	then
	
	ECHELLE_X=5
	if [ $DOSSIER != "Random_task_order" ]
	then
		ECHELLE_X=$((5*NGPU))
	fi
	
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_3.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/DARTS_time.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DARTS_time.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/DARTS_time_no_threshold.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DARTS_time_no_threshold.txt

	# Tracage des GFlops
	gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
	./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_1.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

	#Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GFlops_raw_out_3.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

	#~ # Tracage du temps d'éviction de DDA et de schedule de DDA
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/DARTS_time.txt DARTS_time_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DARTS_time_${MODEL}_${GPU}_${NGPU}GPU.pdf
	#~ mv ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/DARTS_time.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DARTS_time_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ # No threshold
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/DARTS_time_no_threshold.txt DARTS_time_no_threshold_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DARTS_time_no_threshold_${MODEL}_${GPU}_${NGPU}GPU.pdf
	#~ mv ${PATH_STARPU}/starpu/Output_maxime/Data/${DOSSIER}/DARTS_time_no_threshold.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DARTS_time_no_threshold_${MODEL}_${GPU}_${NGPU}GPU.txt
fi
if [ $MODEL == "dynamic_data_aware_no_hfp_profiling" ]
	then
	ECHELLE_X=$((5*NGPU))
	
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/Schedule_time_raw_out.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/Schedule_time_raw_out.txt

	# Tracage du temps global et du temps de schedule all et split
	gcc -o cut_schedule_time_raw_out cut_schedule_time_raw_out.c
	./cut_schedule_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/Schedule_time_raw_out.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt SCHEDULE_TIME_${MODEL}_ipdps ${DOSSIER} ${GPU} ${NGPU}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
if [[ $MODEL == "dynamic_data_aware_compare_threshold" || $MODEL == "dynamic_data_aware_compare_threshold_worse_time" ]]
	then
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_1.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_3.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/DDA_eviction_time.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/DDA_eviction_time.txt
	
	ECHELLE_X=20
	
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

	# Tracage du temps d'éviction et de schedule de DDA
	Rscript ${PATH_R}/R/ScriptR/GF_X.R /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/DDA_eviction_time.txt EVICTION_TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU}
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/EVICTION_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
	mv /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/DDA_eviction_time.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/EVICTION_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
fi
if [ $MODEL == "dynamic_data_aware_compare_threshold_type" ]
	then
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_1.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_3.txt
	
	ECHELLE_X=$((5*NGPU))
	
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
fi
if [ $MODEL == "dynamic_data_aware_compare_choose_best_data_type" ]
	then
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_1.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_1.txt
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt /home/gonthier/starpu/Output_maxime/Data/Matrice_ligne/GFlops_raw_out_3.txt
	
	ECHELLE_X=$((5*NGPU))
	
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
fi
