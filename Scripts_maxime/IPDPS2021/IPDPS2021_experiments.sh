PATH_SIMGRID=$1
PATH_STARPU=$2
PATH_R=$3
./configure --enable-simgrid --disable-mpi --with-simgrid-dir=${PATH_SIMGRID}
sudo make -j 6
sudo make install
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 1
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 2
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 3
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 4
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 10 Random_task_order dynamic_data_aware_no_hfp Attila 1
bash Scripts_maxime/dynamic_data_aware.sh ${PATH_STARPU} ${PATH_R} 10 Random_task_order dynamic_data_aware_no_hfp Attila 2
