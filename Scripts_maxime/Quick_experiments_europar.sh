#!/usr/bin/bash
start=`date +%s`
PATH_SIMGRID=$1
PATH_STARPU=$2
PATH_R=$3
./configure --enable-simgrid --disable-mpi --with-simgrid-dir=${PATH_SIMGRID}/simgrid
sudo make install
sudo make -j4
bash Scripts_maxime/Matrice_ligne/GF_M_MC_NT=225_LRU_BW350.sh ${PATH_STARPU} ${PATH_R} 10
bash Scripts_maxime/Matrice_ligne/GF_NT_MC_LRU_BW350_CM500.sh ${PATH_STARPU} ${PATH_R} 8
echo "Matrix 2D done"
bash Scripts_maxime/Matrice3D/GF_NT_M3D_BW350_CM500.sh ${PATH_STARPU} ${PATH_R} 6
echo "Matrix 3D done"
bash Scripts_maxime/Random_tasks/GF_NT_MC_LRU_BW350_CM500_RANDOMTASKS.sh ${PATH_STARPU} ${PATH_R} 8
echo "Random 2D matrix done"
bash Scripts_maxime/Cholesky/GF_NT_CHO_BW350_CM500.sh ${PATH_STARPU} ${PATH_R} 5
echo "Cholesky done"
end=`date +%s`
runtime=$((end-start))
echo "End of script, it lasted" $((runtime/60))" min "$((runtime%60))" sec."
