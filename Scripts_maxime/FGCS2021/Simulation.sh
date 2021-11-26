#!/usr/bin/bash
PATH_SIMGRID=$1
PATH_STARPU=$2
PATH_R=$3
./configure --enable-simgrid --disable-mpi --with-simgrid-dir=${PATH_SIMGRID}
sudo make -j 6
sudo make install
bash Scripts_maxime/HFP.sh ${PATH_STARPU} ${PATH_R} 10 Matrice3D HFP gemini-1-fgcs 1
