# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL

# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 2000 highest_prio_default_case 2 Cholesky_dependances
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 12 2000 highest_prio_default_case 2 Cholesky_dependances
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 12 2000 highest_prio_default_case 2 Cholesky_dependances
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 12 2000 highest_prio_default_case 2 Cholesky_dependances

#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 2000 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 12 2000 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 12 2000 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 12 2000 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 2880 12 4500 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 2880 12 4500 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 2880 12 4500 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 2880 12 4500 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 3840 12 8000 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 3840 12 8000 opti 8 Cholesky_dependances

#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 32000 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 12 32000 opti 8 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 2880 12 32000 opti 8 Cholesky_dependances

# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 2000 best_ones 6 Cholesky_dependances
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 12 2000 best_ones 6 Cholesky_dependances
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 12 2000 best_ones 6 Cholesky_dependances
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 12 2000 best_ones 6 Cholesky_dependances

# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 2880 12 4500 best_ones 6 Cholesky_dependances

# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 32000 best_ones 6 Cholesky_dependances
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 12 32000 best_ones 6 Cholesky_dependances

# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 2000 best_ones 6 LU
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 12 2000 best_ones 6 LU
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 12 2000 best_ones 6 LU
# bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 12 2000 best_ones 6 LU

if [ $# != 7 ]
then
	echo "Arguments must be: bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL NB_ALGO_TESTE DOSSIER"
	exit
fi

NGPU=$1
TAILLE_TUILE=$2
NB_TAILLE_TESTE=$3
MEMOIRE=$4
MODEL=$5
NB_ALGO_TESTE=$6
DOSSIER=$7
ECHELLE_X=5
START_X=0

scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/Data/${DOSSIER}/GF_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/

scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/Data/${DOSSIER}/DT_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/


gcc -o cut_gflops_raw_out_csv cut_gflops_raw_out_csv.c
./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/GF_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv

gcc -o cut_datatransfers_raw_out_csv cut_datatransfers_raw_out_csv.c
./cut_datatransfers_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU /home/gonthier/starpu/Output_maxime/Data/${DOSSIER}/DT_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv


python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/PlaFRIM-Grid5k/${DOSSIER}/GF_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv GF_PlaFRIM-Grid5k $1 $2 $3 $4 $5 $7

python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/PlaFRIM-Grid5k/${DOSSIER}/DT_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv DT_PlaFRIM-Grid5k $1 $2 $3 $4 $5 $7
