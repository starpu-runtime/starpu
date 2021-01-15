#!/usr/bin/bash
# ./cut_gflops_raw_out NOMBRE_DE_TAILLES_DE_MATRICES NOMBRE_ALGO_TESTE
cd..
#~ ./configure --enable-simgrid --disable-mpi
./configure --enable-simgrid --disable-mpi --with-fxt=/home/gonthier/fxt --with-simgrid-dir=/home/gonthier/simgrid
make install
make -j4
echo "MAKE OK!"
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
truncate -s 0 GFlops_raw_out.txt
ulimit -S -s 50000
NB_ALGO_TESTE=1
NB_TAILLE_TESTE=10
TAILLE=5
#~ FICHIER=Matrice3D/I_M_MC3D_LRU_N=15
#~ FICHIER_R=Matrice3D/I_M_MC3D_LRU_N=15.R
#~ NB_TEST_MEME_SOMMET=1
echo "Random"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	TAILLE=$((TAILLE*i))
	STARPU_SCHED=eager STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*$TAILLE)) -nblocks $TAILLE -iter 1 | tail -n 1 >> GFlops_raw_out.txt
done
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE
