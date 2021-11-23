#!/usr/bin/bash
NAME=$1
PATH_STARPU=$2
PATH_R=$3
./configure --prefix=/home/${NAME}/starpu
make -j 100
make install
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 8 Matrice_ligne HFP 1
bash Scripts_maxime/FGCS2021/Draw.sh 8 Matrice_ligne HFP 1 9 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice_ligne HFP_memory 1
bash Scripts_maxime/FGCS2021/Draw.sh 10 Matrice_ligne HFP_memory 1 9 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 6 Matrice3D HFP 1
bash Scripts_maxime/FGCS2021/Draw.sh 6 Matrice3D HFP 1 9 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 6 Cholesky HFP 1
bash Scripts_maxime/FGCS2021/Draw.sh 6 Cholesky HFP 1 9 ${NAME} ${PATH_STARPU} ${PATH_R}
