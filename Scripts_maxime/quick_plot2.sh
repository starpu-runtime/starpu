# bash Scripts_maxime/quick_plot2.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL

# bash Scripts_maxime/quick_plot2.sh 1 960 12 memory

if [ $# != 5 ]
then
	echo "Arguments must be: bash Scripts_maxime/quick_plot2.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE MODEL"
	exit
fi

echo "Plotting"
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot.csv GF $1 $2 $3 $4 $5
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot_dt.csv DT $1 $2 $3 $4 $5
