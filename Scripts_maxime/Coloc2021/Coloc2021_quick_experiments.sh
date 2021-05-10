PATH_SIMGRID=$1
PATH_STARPU=$2
PATH_R=$3
./configure --enable-simgrid --disable-mpi --with-simgrid-dir=${PATH_SIMGRID}/simgrid
sudo make -j 4
sudo make install
bash Scripts_maxime/GF_Workingset.sh ${PATH_STARPU} ${PATH_R} 6 Matrice3D Workingset_europar Gemini02
bash Scripts_maxime/GF_Workingset.sh ${PATH_STARPU} ${PATH_R} 10 Matrice_ligne Memory_europar Gemini02
