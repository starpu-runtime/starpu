#!/usr/bin/bash
module load linalg/mkl
ulimit -S -s 50000000

NB_TAILLE_TESTE=1
NB_ALGO_TESTE=6
ECHELLE_X=5
START_X=0 
FICHIER_RAW=Output_maxime/GFlops_raw_out_3.txt
truncate -s 0 ${FICHIER_RAW}

STARPU_SCHED=eager ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
STARPU_SCHED=dmdar ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
STARPU_SCHED=mst ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
STARPU_SCHED=cuthillmckee REVERSE=1 ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
STARPU_SCHED=HFP ORDER_U=1 ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
STARPU_SCHED=HFP ORDER_U=1 BELADY=1 ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}

gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} Output_maxime/GFlops.txt

echo "Fin du script"

#~ module load linalg/mkl
#~ libtool --mode=execute gdb --args
#~ STARPU_SCHED=HFP ./examples/mult/sgemm -xy $((960*5)) -nblocks 5 -iter 1

#~ STARPU_SCHED=HFP TASK_STEALING=3 PRINTF=1 MULTIGPU=4 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1

#~ STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila libtool --mode=execute gdb --args ./examples/mult/sgemm -xy $((960*5)) -nblocks 5 -iter 1
