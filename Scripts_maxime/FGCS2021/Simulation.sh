# Lancer depuis starpu/ avec bash Scripts_maxime/FGCS2021/Simulation.sh path_simgrid path_starpu path_r
PATH_SIMGRID=$1
PATH_STARPU=$2
PATH_R=$3
./configure --enable-simgrid --disable-mpi --with-simgrid-dir=${PATH_SIMGRID}
make -j 10
sudo make install
bash Scripts_maxime/HFP.sh ${PATH_STARPU} ${PATH_R} 10 Matrice3DZN HFP gemini-1-fgcs 1
