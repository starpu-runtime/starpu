#!/usr/bin/bash
#bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne UBfactor
#bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne Workingset_hMETIS
#bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice3D Workingset_hMETIS

	#~ //~ Nparts : nombre de paquets.
	#~ //~ UBfactor : 1 - 49, a tester. Déséquilibre autorisé.
	#~ //~ Nruns : 1 - inf, a tester. 1 par défaut. Plus on test plus ce sera bon mais ce sera plus long.
	#~ //~ CType : 1 - 5, a tester. 1 par défaut.
	#~ //~ RType :  1 - 3, a tester. 1 par défaut.
	#~ //~ Vcycle : 1. Sélectionne la meilleure des Nruns. 
	#~ //~ Reconst : 0 - 1, a tester. 0 par défaut. Normalement ca ne devrait rien changer car ca joue juste sur le fait de reconstruire les hyperedges ou non.
	#~ //~ dbglvl : 0. Sert à montrer des infos de debug; Si besoin mettre (1, 2 ou 4).

PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
NB_ALGO_TESTE=2
FICHIER_RAW=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_1.txt
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}
if [ $DOSSIER = "Matrice_ligne" ]
	then
	if [ $MODEL = "UBfactor" ]
	then
		START_X=-1
		ECHELLE_X=2
		echo "############## HMETIS UBfactor ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3" $N "20 1 1 1 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS UBfactor + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3" $N "20 1 1 1 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	if [ $MODEL = "Nruns" ]
	then
		START_X=-1
		ECHELLE_X=2
		echo "############## HMETIS Nruns ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1" $N "1 1 1 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS Nruns + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1" $N "1 1 1 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	if [ $MODEL = "CType" ]
	then
		START_X=0
		ECHELLE_X=1
		echo "############## HMETIS CType ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20" $N "1 1 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS CType + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20" $N "1 1 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	if [ $MODEL = "RType" ]
	then
		START_X=0
		ECHELLE_X=1
		echo "############## HMETIS RType ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1" $N "1 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS RType + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1" $N "1 0 0" > Output_maxime/hMETIS_parameters.txt  
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	if [ $MODEL = "Reconst" ]
	then
		START_X=-1
		ECHELLE_X=1
		echo "############## HMETIS Reconst ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 1" $N "0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS Reconst + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 1" $N "0" > Output_maxime/hMETIS_parameters.txt
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	if [ $MODEL = "Vcycle" ]
	then
		START_X=-1
		ECHELLE_X=1
		echo "############## HMETIS Vcycle ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1" $N "0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS Vcycle + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1" $N "0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*40)) -nblocks 40 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
	fi
	if [ $MODEL = "Workingset_hMETIS" ]
	then
		NB_ALGO_TESTE=7
		START_X=0
		ECHELLE_X=5
		echo "############## HMETIS Working set ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS Working set + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS Working set + TASK_STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}	
		done
		echo "############## HMETIS Working set + HFP + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}	
		done
		echo "############## HMETIS Working set + MST ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=mst HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HMETIS Working set + MST + task stealing ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=mst HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}	
		done
		echo "############## HFP + load balance + task stealing ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP TASK_STEALING=3 LOAD_BALANCE=4 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}	
		done
	fi
fi
if [ $DOSSIER = "Matrice3D" ]
	then
	if [ $MODEL = "Workingset_hMETIS" ]
	then
		NB_ALGO_TESTE=7
		START_X=0
		ECHELLE_X=5
		echo "############## HMETIS Working set ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 | tail -n 1 >> ${FICHIER_RAW}		
		done
		echo "############## HMETIS Working set + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 | tail -n 1 >> ${FICHIER_RAW}		
		done
		echo "############## HMETIS Working set + TASK_STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 | tail -n 1 >> ${FICHIER_RAW}		
		done
		echo "############## HMETIS Working set + HFP + TASK STEALING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP HMETIS=2 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 | tail -n 1 >> ${FICHIER_RAW}		
		done
		echo "############## HMETIS Working set + MST ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=mst HMETIS=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 | tail -n 1 >> ${FICHIER_RAW}		
		done
		echo "############## HMETIS Working set + MST + task stealing ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=mst HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 | tail -n 1 >> ${FICHIER_RAW}		
		done
		echo "############## HFP + load balance + task stealing ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			echo "3 1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			STARPU_SCHED=HFP TASK_STEALING=3 LOAD_BALANCE=4 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 | tail -n 1 >> ${FICHIER_RAW}		
		done
	fi
fi
# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/hMETIS/GF_${MODEL}.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/hMETIS/GF_${MODEL}.txt ${MODEL} ${DOSSIER}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/hMETIS/GF_${MODEL}.pdf
