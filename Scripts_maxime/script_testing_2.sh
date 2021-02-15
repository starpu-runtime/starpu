#!/usr/bin/bash
start=`date +%s`
#~ sudo make -j4
#~ sudo make -j4 -C src/
#~ sudo make -j4 -C examples/
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 50000000

#~ ECHELLE_X=100
#~ START_X=-50
#~ NB_TAILLE_TESTE=9
#~ echo "########## RCM GF_M_CHO ##########"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ M=$((START_X+i*ECHELLE_X))
	#~ STARPU_SCHED=cuthillmckee REVERSE=0 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*20)) -nblocks 20
#~ done
#~ NB_TAILLE_TESTE=7
#~ ECHELLE_X=5
#~ START_X=0
#~ echo "############## RCM GF_NT_CHO ##############"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ N=$((START_X+i*ECHELLE_X))
	#~ STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_BUS_STATS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
#~ done
#~ NB_TAILLE_TESTE=10
#~ ECHELLE_X=50
#~ START_X=0
#~ echo "########## RCM GF_M_M3D ##########"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ M=$((START_X+i*ECHELLE_X))
	#~ STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*15)) -nblocks $((15)) -nblocksz 4 -iter 1
#~ done
#~ NB_TAILLE_TESTE=8
#~ ECHELLE_X=5
#~ START_X=0
#~ echo "############## RCM GF_NT_M3D ##############"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ N=$((START_X+i*ECHELLE_X))
	#~ STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_BUS_STATS=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1
#~ done
#~ NB_TAILLE_TESTE=10
#~ ECHELLE_X=50
#~ START_X=0
#~ echo "########## RCM GF_M_MC ##########"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ M=$((START_X+i*ECHELLE_X))
	#~ STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter 1
#~ done
#~ NB_TAILLE_TESTE=10
#~ ECHELLE_X=5
#~ START_X=0   
#~ echo "############## RCM ##############"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ N=$((START_X+i*ECHELLE_X))
	#~ STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_BUS_STATS=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
#~ done
#~ NB_TAILLE_TESTE=10
#~ ECHELLE_X=50
#~ START_X=0
#~ echo "########## RCM ##########"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ M=$((START_X+i*ECHELLE_X))
	#~ STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter 1
#~ done
NB_TAILLE_TESTE=10
ECHELLE_X=5
START_X=0
echo "############## RCM ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_NTASKS_THRESHOLD=30 RANDOM_DATA_ACCESS=1 SEED=1 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_BANDWIDTH=350 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_BUS_STATS=1 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
done

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
