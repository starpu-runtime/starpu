#!/usr/bin/bash
start=`date +%s`
sudo make -j4
#~ make -j4 -C src/
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 50000000
#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=50 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*5)) -nblocks 5 -iter 1

STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_TASK_PROGRESS=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=50 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*4)) -nblocks 10 -nblocksz 4 -iter 1

#~ STARPU_SCHED=dmdar STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*40)) -nblocks 40 -nblocksz 4 -iter 1

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*10)) -nblocks 10 

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*40)) -nblocks 40 

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=50 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -no-prio -size $((960*10)) -nblocks 10

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*30)) -nblocks 30

#~ STARPU_SCHED=mst PRINTF=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*40)) -nblocks 40

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=1000 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=50 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*10)) -nblocks 10 -no-prio

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60)) "minutes."

#~ Salut, j'ai fais des tests sur cholesky avec HFPU + N=10
#~ cholesky sans threshold et sans prio : 155.9
#~ cholesky sans threshold et avec prio : 201.2
#~ cholesky avec threshold30 et sans prio : 228.2
#~ cholesky avec threshold30 et avec prio : 228.2
#~ cholesky avec threshold30 et sans prio et mémoire basse : Affiche le message de not enough memory on CUDA 0 et ne termine pas
#~ cholesky sans threshold et sans prio et mémoire basse : 128.7
#~ cholesky sans threshold et avec prio et mémoire basse : 128.7
#~ cholesky avec threshold10 et avec prio et mémoire basse : 96.5

#~ Du coup j'ai vraiment l'impression que c'es le threshold qui bloque

