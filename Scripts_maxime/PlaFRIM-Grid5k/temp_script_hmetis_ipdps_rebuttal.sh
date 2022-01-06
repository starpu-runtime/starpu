#  Pour test Cholesky avec temps de schedule
#	bash Scripts_maxime/PlaFRIM-Grid5k/temp_script_hmetis_ipdps_rebuttal.sh

NB_TAILLE_TESTE=15
START_X=0  
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
truncate -s 0 ${FICHIER_RAW}
ulimit -S -s 5000000
HMETIS_APPLI=1
CM=500
TH=10
CP=5
NITER=11
ECHELLE_X=$((5))

echo "############## Matrice2D 2 GPUs ##############"
NGPU=2
ECHELLE_X=$((5*NGPU))
echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt
echo "############## HMETIS + TASK STEALING ##############"
for ((i=1 ; i<=((15)); i++))
	do
	N=$((START_X+i*ECHELLE_X))
	COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP HMETIS=1 HMETIS_N=$((N)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
done

echo "############## Matrice2D 4 GPUs ##############"
NGPU=4
ECHELLE_X=$((5*NGPU))
echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt
echo "############## HMETIS + TASK STEALING ##############"
for ((i=1 ; i<=((15)); i++))
	do
	N=$((START_X+i*ECHELLE_X))
	COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP HMETIS=1 HMETIS_N=$((N)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
done

echo "############## Randomized Matrice2D 2 GPUs ##############"
NGPU=2
ECHELLE_X=5
echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt
echo "############## HMETIS + TASK STEALING ##############"
for ((i=1 ; i<=((13)); i++))
	do
	N=$((START_X+i*ECHELLE_X))
	RANDOM_TASK_ORDER=1 SEED=$((i)) COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP HMETIS=1 HMETIS_N=$((N)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
done

#~ echo "############## Courbes rebuttal 4 GPUs ##############"
#~ NGPU=4
#~ echo "############## CHOLESKY ##############"
#~ echo "############## HMETIS_N=$((N)) HMETIS + TASK STEALING ##############"
#~ for ((i=1 ; i<=((14)); i++))
	#~ do 
	#~ N=$((START_X+i*ECHELLE_X))
	#~ COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=$((HMETIS_APPLI)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SCHED_READY=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
#~ done

#~ echo "############## SPARSE ##############"
#~ ECHELLE_X=$((50))
#~ SPARSE=2
#~ echo "############## HMETIS + TASK STEALING ##############"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ N=$((START_X+i*ECHELLE_X))
	#~ COUNT_DO_SCHEDULE=1 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=$((HMETIS_APPLI)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
#~ done

#~ echo "############## SPARSE INFINIE ##############"
#~ CM=0
#~ echo "############## HMETIS + TASK STEALING ##############"
#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	#~ do 
	#~ N=$((START_X+i*ECHELLE_X))
	#~ COUNT_DO_SCHEDULE=1 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=$((HMETIS_APPLI)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
#~ done

