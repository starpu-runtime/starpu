# bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_Rebuttal_test.sh 10 Matrice3D dynamic_data_aware_no_hfp 1
# bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_Rebuttal_test.sh 8 Matrice3D dynamic_data_aware_no_hfp 2
# bash Scripts_maxime/PlaFRIM-Grid5k/Recup_data_Rebuttal_test.sh 5 Cholesky dynamic_data_aware_no_hfp 2

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0
GPU=gemini-1-fgcs
PATH_R=/home/gonthier/these_gonthier_maxime/Starpu
PATH_STARPU=/home/gonthier
SPARSE=""
ECHELLE_X=$((5*NGPU))
NITER=11
NB_ALGO_TESTE=21
NB_ALGO_TESTE=12
#~ if [ $NGPU != 1 ]
#~ then
	#~ NB_ALGO_TESTE=$((NB_ALGO_TESTE+1))
#~ fi

if [ $DOSSIER == "Matrice3D" ]
then
	FICHIER_GF=GF_HFP_M3D_${NGPU}GPU_test.txt
fi
if [ $DOSSIER == "Cholesky" ]
then
	FICHIER_GF=GF_HFP_CHO_${NGPU}GPU_test.txt
fi
if [ $DOSSIER == "Sparse" ]
then
	FICHIER_GF=GF_HFP_SPARSE_${NGPU}GPU_test.txt
fi

#~ scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/${FICHIER_GF} /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF}

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
gcc -o cut_gflops_raw_out_csv cut_gflops_raw_out_csv.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF} ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_test_${GPU}_${NGPU}GPU.txt
./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/${FICHIER_GF} ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_test_${GPU}_${NGPU}GPU.csv
# Plot FULL
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_test_${GPU}_${NGPU}GPU.csv
mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_test_${GPU}_${NGPU}GPU_FULL.pdf

#~ # Plot Short
#~ python3 /home/gonthier/these_gonthier_maxime/Code/Plot_shorten.py ${PATH_R}/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_test_${GPU}_${NGPU}GPU.csv
#~ mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_test_${GPU}_${NGPU}GPU.pdf
