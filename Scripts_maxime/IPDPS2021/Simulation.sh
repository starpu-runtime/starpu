# Lancer depuis starpu/ avec bash Scripts_maxime/IPDPS2021/Simulation.sh path_simgrid path_starpu path_r
PATH_SIMGRID=$1
PATH_STARPU=$2
PATH_R=$3
./configure --enable-simgrid --disable-mpi --with-simgrid-dir=${PATH_SIMGRID}
sudo make -j 10
sudo make install
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 12 Matrice3D dynamic_data_aware_no_hfp gemini-1-fgcs 4
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-fgcs 2
