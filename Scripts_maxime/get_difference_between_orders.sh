#!/usr/bin/bash
#~ bash get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Diff_HFP_HEFT_BW350_CM500
PATH_STARPU=$1
PATH_R=$2
TAILLE_TESTE=$3
DOSSIER=$4
FICHIER=$5
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000
N=$((TAILLE_TESTE))
STARPU_SCHED=modular-heft-HFP STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1

# Tracage des GFlops
gcc -o get_difference_between_orders get_difference_between_orders.c
./get_difference_between_orders Output_maxime/Task_order_HFP.txt Output_maxime/Task_order_effective.txt ${PATH_R}/R/Data/${DOSSIER}/Difference_between_orders/${FICHIER:0}.txt
Rscript ${PATH_R}/R/ScriptR/${DOSSIER}/Difference_between_orders/Diff_HFP_HEFT_BW350_CM500.R ${PATH_R}/R/Data/${DOSSIER}/Difference_between_orders/${FICHIER}.txt
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Difference_between_orders/${FICHIER:0}.pdf
