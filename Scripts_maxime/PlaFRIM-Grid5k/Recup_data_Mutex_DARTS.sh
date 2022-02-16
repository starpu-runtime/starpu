# 	bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_Mutex_DARTS.sh 12 Matrice_ligne mutex_darts 4
#	bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_Mutex_DARTS.sh 3 Matrice3D mutex_darts 4
#	bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_Mutex_DARTS.sh 3 Cholesky mutex_darts 4
#	bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_Mutex_DARTS.sh 7 Cholesky mutex_darts 2

PATH_STARPU=/home/gonthier/
PATH_R=/home/gonthier/these_gonthier_maxime/Starpu/
NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0
GPU=gemini-1-fgcs
PATH_R=/home/gonthier/these_gonthier_maxime/Starpu
PATH_STARPU=/home/gonthier
ECHELLE_X=$((5*NGPU))
NITER=11

if [ $DOSSIER == "Matrice_ligne" ]
then
	NB_ALGO_TESTE=2
	
	FICHIER_GF=GF_mutex_M2D_${NGPU}GPU.txt
	FICHIER_CONFLITS=Conflits_mutex_M2D_${NGPU}GPU.txt

	FICHIER_GF_REFINED=GF_refined_mutex_M2D_${NGPU}GPU.txt
	FICHIER_CONFLITS_REFINED=Nb_conflit_donnee_refined_mutex_M2D_${NGPU}GPU.txt
	FICHIER_CONFLITS_CRITIQUE_REFINED=Nb_conflit_donnee_critique_refined_mutex_M2D_${NGPU}GPU.txt
	
	FICHIER_GF_LINEAR=GF_linear_mutex_M2D_${NGPU}GPU.txt
fi
if [ $DOSSIER == "Matrice3D" ]
then
	NB_ALGO_TESTE=4
	
	FICHIER_GF=GF_mutex_M3D_${NGPU}GPU.txt
	FICHIER_CONFLITS=Conflits_mutex_M3D_${NGPU}GPU.txt

	FICHIER_GF_REFINED=GF_refined_mutex_M3D_${NGPU}GPU.txt
	FICHIER_CONFLITS_REFINED=Nb_conflit_donnee_refined_mutex_M3D_${NGPU}GPU.txt
	FICHIER_CONFLITS_CRITIQUE_REFINED=Nb_conflit_donnee_critique_refined_mutex_M3D_${NGPU}GPU.txt
	
	FICHIER_GF_LINEAR=GF_linear_mutex_M3D_${NGPU}GPU.txt
fi
if [ $DOSSIER == "Cholesky" ]
then
	NB_ALGO_TESTE=6
	
	FICHIER_GF=GF_mutex_CHO_${NGPU}GPU.txt
	FICHIER_CONFLITS=Conflits_mutex_CHO_${NGPU}GPU.txt

	FICHIER_GF_REFINED=GF_refined_mutex_CHO_${NGPU}GPU.txt
	FICHIER_CONFLITS_REFINED=Nb_conflit_donnee_refined_mutex_CHO_${NGPU}GPU.txt
	FICHIER_CONFLITS_CRITIQUE_REFINED=Nb_conflit_donnee_critique_refined_mutex_CHO_${NGPU}GPU.txt
	
	FICHIER_GF_LINEAR=GF_linear_mutex_CHO_${NGPU}GPU.txt
fi

#~ # Get data files from ssh
#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/${FICHIER_GF_REFINED} /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF_REFINED}
#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/${FICHIER_CONFLITS_REFINED} /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_CONFLITS_REFINED}
#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/${FICHIER_CONFLITS_CRITIQUE_REFINED} /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_CONFLITS_CRITIQUE_REFINED}

if [ $DOSSIER == "Cholesky" ]
then
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops_raw_out_3.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DT_mutex_CHO_${NGPU}GPU.txt
	cat /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DT_linear_mutex_CHO_${NGPU}GPU.txt /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DT_mutex_CHO_${NGPU}GPU.txt > /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DT_CHO_${NGPU}GPU.txt
fi

#~ # Concaténation
#~ cat /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF_LINEAR} /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF_REFINED} > /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF}
#~ cat /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_CONFLITS_REFINED} /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_CONFLITS_CRITIQUE_REFINED} > /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_CONFLITS}	

#~ # Tracage des GFlops
#~ gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
#~ gcc -o cut_gflops_raw_out_csv cut_gflops_raw_out_csv.c
#~ ./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF} ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ ./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF} ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv
#~ python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv
#~ mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ # Plot conflits de données
#~ ./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_CONFLITS} ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_conflits_${GPU}_${NGPU}GPU.csv
#~ python3 /home/gonthier/these_gonthier_maxime/Code/Plot_conflits.py ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_conflits_${GPU}_${NGPU}GPU.csv
#~ mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_conflits_${GPU}_${NGPU}GPU.pdf

# Plot des transferts de données
if [ $DOSSIER == "Cholesky" ]
then
	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DT_CHO_${NGPU}GPU.txt ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_mutex_CHO_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_mutex_CHO_${NGPU}GPU.txt DT_mutex_darts ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/DT_mutex_CHO_${NGPU}GPU.pdf
fi
