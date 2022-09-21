#!/usr/bin/bash
#	bash Scripts_maxime/PlaFRIM-Grid5k/dmdar.sh 1
#	oarsub -t exotic -p "network_address in ('gemini-1.lyon.grid5000.fr')" -l walltime=14:00:00 -r '2022-09-14 19:00:00' "bash Scripts_maxime/FGCS2021/Experiments.sh mgonthier"
NGPU=$1
START_X=0 
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
ulimit -S -s 500000000
truncate -s 0 ${FICHIER_RAW}

CM=500
TH=10
CP=5
NITER=11


ECHELLE_X=5
NB_TAILLE_TESTE=18
echo "############## Dmdar WRITEBACK 0 ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=0 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "############## Dmdar WRITEBACK 1 ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
done
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_dmdar_writeback_M2D.txt

truncate -s 0 ${FICHIER_RAW}
ECHELLE_X=2
NB_TAILLE_TESTE=10
echo "############## Dmdar WRITEBACK 0 ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	ZN=$((N))
	STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=0 COUNT_DO_SCHEDULE=0 INVALIDATE_C_TILE=0 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -z $((960*ZN)) -nblocksz $((ZN)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "############## Dmdar WRITEBACK 1 ##############"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	N=$((START_X+i*ECHELLE_X))
	ZN=$((N))
	STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 COUNT_DO_SCHEDULE=0 INVALIDATE_C_TILE=0 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -z $((960*ZN)) -nblocksz $((ZN)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
done
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_dmdar_writeback_M3DZN.txt
