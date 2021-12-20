#~ bash Generate_hMETIS_input.sh Cholesky

start=`date +%s`

ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
DOSSIER=$1
NGPU=4
HOST="gemini-1-fgcs"

echo "############## Pour générer des data pour hMETIS ##############"

for ((i=1 ; i<=15; i++))
	do 
	N=$((5*i))
	echo "############## N = $((N)) ##############"
	echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
	STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
	mv Output_maxime/input_hMETIS.txt.part.${NGPU} Output_maxime/Data/input_hMETIS/${NGPU}GPU_${DOSSIER}/input_hMETIS_N${N}.txt
done

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
