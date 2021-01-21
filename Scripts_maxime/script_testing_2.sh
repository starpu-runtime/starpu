#!/usr/bin/bash
#~ make -j4 -C src/
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
#~ ulimit -S -s 50000
#~ echo "MATRICE3D"
#~ STARPU_SCHED=dmdar STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=50 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1
#~ end=`date +%s`
#~ runtime=$((end-start))
#~ echo "Fin du script, l'execution a durée" $((runtime/60)) "minutes."

#!/usr/bin/bash
start=`date +%s`
#~ make -j4 -C src/
sudo make -j4
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000
#~ echo "MATRICE3D"
#~ STARPU_SCHED=mst PRINTF=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*5)) -z $((960*5)) -nblocks 5 -iter 1

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*5)) -nblocks 5 -iter 1

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=1000 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1

#~ STARPU_SCHED=HFP PRINTF=2 ORDER_U=0 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*5)) -z $((960*5)) -nblocks 5 -iter 1

STARPU_SCHED=HFP PRINTF=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*5)) -nblocks 5

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60)) "minutes."
