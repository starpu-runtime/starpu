NAME=$1
PATH_R=$3
PATH_STARPU=$2
module load linalg/mkl
./configure --prefix=/home/${NAME}/starpu
make -j 100
make install
echo "INITIALISATION OK"
export STARPU_PERF_MODEL_DIR=/home/${NAME}/starpu/perfmodels/sampling
bash Scripts_maxime/PlaFRIM/GF_Workingset.sh 9 Matrice_ligne
bash Scripts_maxime/PlaFRIM/Draw.sh ${PATH_STARPU} ${PATH_R} Matrice_ligne Workingset_PlaFRIM Gemini02 Grid5000 ${NAME}
