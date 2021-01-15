#!/usr/bin/bash
# ./cut_gflops_raw_out NOMBRE_DE_TAILLES_DE_MATRICES NOMBRE_ALGO_TESTE

cd..
#~ ./configure --enable-simgrid --disable-mpi --with-simgrid-dir=/home/gonthier/simgrid
#~ make install
#~ make -j4
#~ echo "MAKE OK!"
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
#~ truncate -s 0 GFlops_raw_out.txt
make -j4 -C src/
ulimit -S -s 50000
#~ NB_ALGO_TESTE=1
#~ NB_TAILLE_TESTE=1
#~ TAILLE=500
#~ FICHIER=GF_M_MC_NT=225_LRU_BW=350
echo "MST"
STARPU_SCHED=mst STARPU_LIMIT_BANDWIDTH=350 PRINTF=1 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*2)) -nblocks 2 -iter 1
#~ echo "HFP"
#~ ALGO_USED=5 STARPU_SCHED=AATO STARPU_NTASKS_THRESHOLD=30 HILBERT=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 PRINTF=1 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*2)) -nblocks 2 -iter 1
