#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 2
#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 3
#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 4

#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice3D dynamic_data_aware_no_hfp 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice3D dynamic_data_aware_no_hfp 2

#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 8 Cholesky dynamic_data_aware_no_hfp 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Cholesky dynamic_data_aware_no_hfp 2

#  Vraiment pour le rebuttal:
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Cholesky dynamic_data_aware_no_hfp 4
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Sparse dynamic_data_aware_no_hfp 4
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Sparse_mem_infinite dynamic_data_aware_no_hfp 4

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0  
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
truncate -s 0 ${FICHIER_RAW}
ulimit -S -s 5000000
CM=500
TH=10
CP=5
NITER=11
ECHELLE_X=$((5*NGPU))
if [ $DOSSIER = "Matrice_ligne" ]
then
	HMETIS_APPLI=3
fi
if [ $DOSSIER = "Matrice3D" ]
then
	HMETIS_APPLI=5
fi
if [ $DOSSIER = "Cholesky" ]
then
	HMETIS_APPLI=6
	if [ $NGPU = 4 ]
	then
		ECHELLE_X=$((5))
	fi
fi
if [ $DOSSIER = "Sparse" ]
then
	HMETIS_APPLI=3
	SPARSE=2
	ECHELLE_X=$((50))
fi
if [ $DOSSIER = "Sparse_mem_infinite" ]
then
	HMETIS_APPLI=3
	SPARSE=2
	ECHELLE_X=$((50))
	CM=0
fi

if [ $MODEL = "dynamic_data_aware_no_hfp_no_mem_limit" ]
then
	NB_ALGO_TESTE=4
	TAILLE_TUILE=1920
	echo "NO HFP and no mem limit and tile of size" ${TAILLE_TUILE} "and" ${NGPU} "GPU(s)"
	if [ $DOSSIER = "Matrice_ligne" ]
	then
		NITER=3
		ECHELLE_X=$((100))
	    echo "############## Modular eager prefetching ##############"
	    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		    do 
		    N=$((START_X+i*ECHELLE_X))
		    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
	    done
	    echo "############## Dmdar ##############"
	    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		    do 
		    N=$((START_X+i*ECHELLE_X))
		    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
	    done
	    echo "############## Dynamic data aware TH30 Pop best data ##############"
	    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		    do 
		    N=$((START_X+i*ECHELLE_X))
		    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
	    done
	    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
	    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		    do 
		    N=$((START_X+i*ECHELLE_X))
		    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
	    done
	fi
	#~ if [ $DOSSIER = "Matrice3D" ]
	#~ then
		#~ ECHELLE_X=$((50))
		#~ echo "############## Modular eager prefetching ##############"
		#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			#~ do 
			#~ N=$((START_X+i*ECHELLE_X))
			#~ STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		#~ done
		#~ echo "############## Dmdar ##############"
		#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			#~ do 
			#~ N=$((START_X+i*ECHELLE_X))
			#~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		#~ done
		#~ echo "############## Dynamic data aware TH30 Pop best data + EVICTION + 3D ##############"
		#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			#~ do 
			#~ N=$((START_X+i*ECHELLE_X))
			#~ APP=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		#~ done
	#~ fi
fi
if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "dynamic_data_aware" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=7
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s`
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFPUR ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED_READY=1 STARPU_SCHED=HFP SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		else
			echo "Avec HFP et plus de 1 GPU"
			ECHELLE_X=$((5*NGPU))
		    NB_ALGO_TESTE=6
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFPUR + load balance and task stealing TH30 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED_READY=1 STARPU_SCHED=HFP SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 TASK_STEALING=3 MULTIGPU=4 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HMETIS_N=$((N)) HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/HMETIS_N=$((N)) HMETIS_parameters.txt
			    start=`date +%s` 
			    STARPU_SCHED_READY=1 STARPU_SCHED=HFP HMETIS_N=$((N)) HMETIS=3 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    THRESHOLD=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    THRESHOLD=0 PRINT_TIME=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
	if [[ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ]] || [[ $MODEL = "dynamic_data_aware_no_hfp" ]]
	then
		NITER=11
		ECHELLE_X=$((5*NGPU))
		if [[ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ]]
		then
			ECHELLE_X=$((50*NGPU))
			NITER=3
		fi
		NB_ALGO_TESTE=8
		if [ $NGPU != 1 ]
		then
		    NB_ALGO_TESTE=9
		fi
		echo "############## Modular eager prefetching ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dmdar ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		if [ $NGPU != 1 ]
		then
		    echo "############## HMETIS_N=$((N)) HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED_READY=1 STARPU_SCHED=HFP HMETIS_N=$((N)) HMETIS=3 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
		echo "############## Dynamic data aware ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic data aware + EVICTION ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_TIME=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic data aware + EVICTION + TH ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_TIME=0 THRESHOLD=1 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic data aware + EVICTION + SIMULATE MEMORY 1 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SIMULATE_MEMORY=1 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic data aware + EVICTION + CHOOSE_BEST_DATA_FROM 1 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_TIME=0 CHOOSE_BEST_DATA_FROM=1 APP=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic data aware + EVICTION + CHOOSE_BEST_DATA_FROM 1 + SIMULATE_MEMORY=1 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SIMULATE_MEMORY=1 CHOOSE_BEST_DATA_FROM=1 APP=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
	fi
	if [ $MODEL = "dynamic_data_aware_no_hfp_no_mem_limit" ]
	then
		TAILLE_TUILE=1920
		NITER=1
		ECHELLE_X=$((100))
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=4
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Random_task_order" ]
then
	NITER=11
	if [[ $MODEL = "dynamic_data_aware_no_hfp" ]] || [[ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ]]
	then
		ECHELLE_X=$((5*NGPU))
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=6
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		else
		    NB_ALGO_TESTE=7
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
				start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## HMETIS_N=$((N)) HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED_READY=1 STARPU_SCHED=HFP HMETIS_N=$((N)) HMETIS=3 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=0 STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		 fi
	fi
fi
if [ $DOSSIER = "Matrice3D" ]
then
	NB_ALGO_TESTE=5
	if [ $NGPU != 1 ]
	then
		NB_ALGO_TESTE=6
	fi
	echo "############## Modular eager prefetching ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED=modular-eager-prefetching SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## Dmdar ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED=dmdar SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	if [ $NGPU != 1 ]
	then
		echo "############## HMETIS + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED_READY=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=$((HMETIS_APPLI)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	fi
	echo "############## DARTS + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS 3D + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=1 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS 3D + LUF + THRESHOLD 2 + FROMMEM + SIMMEM ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=1 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
fi
if [ $DOSSIER = "Cholesky" ]
then
	NB_ALGO_TESTE=5
	if [ $NGPU != 1 ]
	then
		NB_ALGO_TESTE=6
	fi
	echo "############## Modular eager prefetching ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED=modular-eager-prefetching SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## Dmdar ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		if [ $N > 50 ]
		then
			TH=0
			CP=0
		fi
		STARPU_SCHED=dmdar SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	done
	TH=10
	CP=5
	if [ $NGPU != 1 ]
	then
		echo "############## HMETIS_N=$((N)) HMETIS + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SCHED_READY=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS_N=$((N)) HMETIS + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=$((HMETIS_APPLI)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SCHED_READY=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	echo "############## DARTS + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS 3D + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=1 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS 3D + LUF + THRESHOLD 2 + FROMMEM + SIMMEM ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=1 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
	done
fi
if [ $DOSSIER = "Sparse" ]
then
	SPARSE=2
	NB_ALGO_TESTE=4
	if [ $NGPU != 1 ]
	then
		NB_ALGO_TESTE=5
	fi
	echo "############## Modular eager prefetching ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=modular-eager-prefetching SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## Dmdar ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dmdar SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	if [ $NGPU != 1 ]
	then
		echo "############## HMETIS + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=$((HMETIS_APPLI)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	echo "############## DARTS + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS + LUF + THRESHOLD 2 + FROMMEM + SIMMEM ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=0 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 NATURAL_ORDER=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
fi
if [ $DOSSIER = "Sparse_mem_infinite" ]
then
	SPARSE=2
	NB_ALGO_TESTE=5
	if [ $NGPU != 1 ]
	then
		NB_ALGO_TESTE=6
	fi
	echo "############## Modular eager prefetching ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=modular-eager-prefetching SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## Dmdar ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dmdar SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	if [ $NGPU != 1 ]
	then
		echo "############## HMETIS + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=$((HMETIS_APPLI)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	echo "############## DARTS + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS + LUF + TH2 ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS + LUF + THRESHOLD 2 + FROMMEM + SIMMEM ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=0 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 NATURAL_ORDER=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
fi
