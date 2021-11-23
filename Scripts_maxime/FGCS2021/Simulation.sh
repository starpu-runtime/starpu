#!/usr/bin/bash
PATH_SIMGRID=$1
PATH_STARPU=$2
PATH_R=$3
./configure --enable-simgrid --disable-mpi --with-simgrid-dir=${PATH_SIMGRID}
sudo make -j 6
sudo make install
bash Scripts_maxime/HFP.sh ${PATH_STARPU} ${PATH_R} 13 Random_tasks HFP gemini-1-fgcs 1
bash Scripts_maxime/HFP.sh ${PATH_STARPU} ${PATH_R} 13 Random_task_order HFP gemini-1-fgcs 1
