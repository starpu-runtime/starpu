#!/usr/bin/bash
start=`date +%s`
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000
NB_ALGO_TESTE=8
NB_TAILLE_TESTE=10
ECHELLE_X=5
START_X=0
DOSSIER=Random_tasks
FICHIER=GF_NT_MC_LRU_BW350_CM500_RANDOMTASKS
FICHIER_DT=DT_NT_MC_LRU_BW350_CM500_RANDOMTASKS
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
FICHIER_BUS=Output_maxime/BUS_STATS_1.txt
truncate -s 0 ${FICHIER_RAW:0}
truncate -s 0 ${FICHIER_RAW_DT:0}
truncate -s 0 ${FICHIER_BUS:0}
echo "############## Random_order ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=random_order STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" RANDOM_DATA_ACCESS=1 SEED=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
echo "############## Dmdar ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=dmdar STARPU_LIMIT_BANDWIDTH=350 STARPU_CUDA_PIPELINE=4 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" RANDOM_DATA_ACCESS=1 SEED=1 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
echo "############## HFP ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_CUDA_PIPELINE=4 ORDER_U=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
echo "############## HFP U ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" STARPU_LIMIT_CUDA_MEM=500 RANDOM_TASK_ORDER=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
echo "############## MST ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=mst STARPU_NTASKS_THRESHOLD=30 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
echo "############## CM ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=cuthillmckee STARPU_NTASKS_THRESHOLD=30 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
echo "############## HFP BELADY ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 BELADY=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_CUDA_PIPELINE=4 ORDER_U=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
echo "############## HFP U BELADY ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 BELADY=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
	sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
done
gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW_DT:0} ../these_gonthier_maxime/Starpu/R/Data/${DOSSIER}/${FICHIER_DT:0}.txt
Rscript /home/gonthier/these_gonthier_maxime/Starpu/R/ScriptR/${DOSSIER}/${FICHIER_DT:0}.R
mv /home/gonthier/starpu/Rplots.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/${DOSSIER}/${FICHIER_DT:0}.pdf
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW:0} ../these_gonthier_maxime/Starpu/R/Data/${DOSSIER}/${FICHIER:0}.txt
Rscript /home/gonthier/these_gonthier_maxime/Starpu/R/ScriptR/${DOSSIER}/${FICHIER:0}.R
mv /home/gonthier/starpu/Rplots.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/${DOSSIER}/${FICHIER:0}.pdf
end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
