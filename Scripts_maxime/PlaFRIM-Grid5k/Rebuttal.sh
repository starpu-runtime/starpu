#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 5 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice3D dynamic_data_aware_no_hfp 2
#	Attention au tracage, les blocs sont de taille 1920

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0  
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}

CM=500

TH=10
CP=5

NITER=11

#~ Tuiles 7680 -> 5 * N15

#~ Attention pour NGPU > 1 faut sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
NCOMBINAISONS=4
if [ NGPU != 1]
then
	NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))
fi

if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "dynamic_data_aware_no_hfp_no_mem_limit" ]
	then
		NITER=2
		TAILLE_TUILE=7680
		ECHELLE_X=$((15*NGPU))
			echo "NO HFP and no mem limit and tuile of size" ${TAILLE_TUILE}
		    NB_ALGO_TESTE=3
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
	fi
fi
if [ $DOSSIER = "Matrice3D" ]
then
	NITER=11
	if [ $MODEL = "dynamic_data_aware_no_hfp" ]
	then
		ECHELLE_X=$((5*NGPU))
		    NB_ALGO_TESTE=5
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware + EVICTION + READY + variante 3D ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    APP=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
	fi
fi
if [ $DOSSIER = "Cholesky" ]
then
	if [ $MODEL = "dynamic_data_aware_no_hfp" ]
	then
		ECHELLE_X=$((5*NGPU))
		    NB_ALGO_TESTE=5
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware + EVICTION + READY + variante 3D ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    APP=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
	fi
fi

