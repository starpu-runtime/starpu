echo "Plotting"
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot.csv GF $1
mv plot.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/quick_plot.pdf
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py /home/gonthier/these_gonthier_maxime/Starpu/R/Data/quick_plot_dt.csv DT $1
mv plot.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/quick_plot_dt.pdf