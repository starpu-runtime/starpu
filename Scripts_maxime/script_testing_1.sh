#!/usr/bin/bash
#~ bash Scripts_maxime/script_testing_1.sh

start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
make -C src/ -j 6
#~ make -j 6

#~ libtool --mode=execute strace ./examples/mult/sgemm

#~ libtool --mode=execute strace -o /tmp/log ./examples/mult/sgemm

#~ srun --exclusive -C sirocco21 --pty bash Scripts_maxime/task_stealing.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne task_stealing -i

#~ bash Scripts_maxime/task_stealing.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne task_stealing

#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1
#~ ./examples/mult/sgemm -3d -xyz $((960*N)) -nblocks $((N)) -nblocksz $((N)) -iter 1
#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5

#~ libtool --mode=execute gdb --args 
#~ r backtrace l p

#~ HOST="attila"
#~ ORDO="dynamic-outer"
#~ N=7
#~ CUDAMEM=500
#~ NGPU=1
#~ BW=350

#~ SEED=1 STARPU_SCHED=${ORDO} EVICTION_STRATEGY_DYNAMIC_OUTER=0 DATA_POP_POLICY=1 PRINTF=1 PRINT_N=$((N)) STARPU_GENERATE_TRACE=0 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CUDAMEM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 

#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 1

#~ watch *(int *)0x555555876ab0 task 0x555555b20da0

#~ watch *(int *)0x5555558771c0
#~ c
#~ c
#~ disable b 1

#~ watch *(int *)0x5555558c2820
#~ watch *(int *)0x5555558c2820

#~ N=30
#~ NGPU=2
#~ ORDO="dynamic-data-aware"
#~ BW=350*NGPU
#~ CM=500
#~ EVICTION=1
#~ READY=0
#~ HOST=attila
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 5000000


		    echo "############## HMETIS + TASK STEALING ##############"
		    NGPU=8
		    ECHELLE_X=$((5*NGPU))
		    START_X=0
		    BW=350*NGPU
		    CM=500
		    for ((i=10; i<=10; i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
			    mv Output_maxime/input_hMETIS.txt.part.${NGPU} Output_maxime/Data/input_hMETIS/${NGPU}GPU/input_hMETIS_N${N}.txt
		    done

#~ SEED=1 PRINTF=0 STARPU_SCHED=${ORDO} STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila valgrind -s ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 2
 
#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 1

#~ libtool --mode=execute gdb --args
#~ setarch linux64 -R libtool --mode=execute gdb --args

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
