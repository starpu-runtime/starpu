#!/usr/bin/bash
start=`date +%s`
ulimit -S -s 5000000
#~ make -C src/ -j 100

N=$1
NGPU=4
#~ ORDO="dynamic-data-aware"
#~ ORDO="dmdar"
#~ ORDO="eager"
#~ BW=10726
CM=500
EVICTION=0
#~ EVICTION=1
#~ READY=0
READY=1
TH=10
CP=5
#~ HOST="gemini-1-ipdps"
#~ HOST="gemini-2-ipdps"
#~ HOST="attila"
#~ export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
#~ NB_TAILLE_TESTE=2
#~ FICHIER_BUS=Output_maxime/GFlops_raw_out_2.txt
#~ FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
#~ truncate -s 0 ${FICHIER_BUS}
#~ truncate -s 0 ${FICHIER_RAW_DT}
#~ NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))


STARPU_SCHED=HFP HMETIS=3 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 11

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
