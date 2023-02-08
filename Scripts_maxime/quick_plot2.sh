# bash Scripts_maxime/quick_plot2.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL

# bash Scripts_maxime/quick_plot2.sh 1 960 12 memory

if [ $# != 5 ]
then
	echo "Arguments must be: bash Scripts_maxime/quick_plot2.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL"
	exit
fi

NGPU=$1
TAILLE_TUILE=$2
NB_TAILLE_TESTE=$3
MEMOIRE=$4
MODEL=$5

echo "Plotting"
mv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot.csv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/Cholesky_dependances/GF_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/Cholesky_dependances/GF_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv GF $1 $2 $3 $4 $5

mv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot_dt.csv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/Cholesky_dependances/DT_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/Cholesky_dependances/DT_${MODEL}_${TAILLE_TUILE}_${NGPU}GPU_${MEMOIRE}Mo.csv DT $1 $2 $3 $4 $5
