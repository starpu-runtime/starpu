NAME=$1
module load linalg/mkl
./configure --prefix=/home/${NAME}/starpu
make -j 100
make install
echo "INITIALISATION OK"
export STARPU_PERF_MODEL_DIR=/home/${NAME}/starpu/perfmodels/sampling
bash Scripts_maxime/PlaFRIM/GF_Workingset.sh 9 Matrice_ligne
