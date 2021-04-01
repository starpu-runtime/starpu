#!/usr/bin/bash
#~ M2D
#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Diff_HFP_HEFT_BW350_CM500_1GPU HEFT 1
#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 35 Matrice_ligne Diff_HFP_HEFT_BW350_CM500_3GPU HEFT 3
#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Diff_HFP_HFPR_BW350_CM500_1GPU HFPR 1
#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Diff_HFP_HFPR_BW350_CM500_3GPU HFPR 3

#~ M3D
#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice3D Diff_HFP_HEFT_BW350_CM500_1GPU_M3D HEFT 1
#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice3D Diff_HFP_HEFT_BW350_CM500_3GPU_M3D HEFT 3

PATH_STARPU=$1
PATH_R=$2
TAILLE_TESTE=$3
DOSSIER=$4
FICHIER=$5
MODEL=$6
GPU=$7
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000
N=$((TAILLE_TESTE))
if [ $DOSSIER = "Matrice_ligne" ]
	then
	NT=$((N*N))
	if [ $MODEL = "HFPR" ]
		then
		if [ $GPU == "1" ]
			then
			STARPU_SCHED=HFP READY=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 PRINTF=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
		fi
		if [ $GPU == "3" ]
			then
			STARPU_SCHED=HFP READY=1 MULTIGPU=4 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 PRINTF=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
			cat "Output_maxime/Task_order_HFP_1" >> "Output_maxime/Task_order_HFP_0"
			cat "Output_maxime/Task_order_HFP_2" >> "Output_maxime/Task_order_HFP_0"
			cat "Output_maxime/Task_order_effective_1" >> "Output_maxime/Task_order_effective_0"
			cat "Output_maxime/Task_order_effective_2" >> "Output_maxime/Task_order_effective_0"
		fi
	fi
	if [ $MODEL = "HEFT" ]
		then
		if [ $GPU == "1" ]
			then
			STARPU_SCHED=modular-heft-HFP MODULAR_HEFT_HFP_MODE=2 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 PRINTF=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
		fi
		if [ $GPU == "3" ]
			then
			STARPU_SCHED=modular-heft-HFP MODULAR_HEFT_HFP_MODE=2 MULTIGPU=4 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 PRINTF=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
			cat "Output_maxime/Task_order_HFP_1" >> "Output_maxime/Task_order_HFP_0"
			cat "Output_maxime/Task_order_HFP_2" >> "Output_maxime/Task_order_HFP_0"
			cat "Output_maxime/Task_order_effective_1" >> "Output_maxime/Task_order_effective_0"
			cat "Output_maxime/Task_order_effective_2" >> "Output_maxime/Task_order_effective_0"
		fi
	fi
fi
if [ $DOSSIER = "Matrice3D" ]
	then
	NT=$((N*N*4))
	if [ $MODEL = "HFPR" ]
		then
		if [ $GPU == "1" ]
			then
			echo "ee"
		fi
		if [ $GPU == "3" ]
			then
			echo "ee"
		fi
	fi
	if [ $MODEL = "HEFT" ]
		then
		if [ $GPU == "1" ]
			then
			STARPU_SCHED=modular-heft-HFP PRINTF=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1
		fi
		if [ $GPU == "3" ]
			then
			STARPU_SCHED=modular-heft-HFP MULTIGPU=4 MODULAR_HEFT_HFP_MODE=2 PRINTF=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1
			cat "Output_maxime/Task_order_HFP_1" >> "Output_maxime/Task_order_HFP_0"
			cat "Output_maxime/Task_order_HFP_2" >> "Output_maxime/Task_order_HFP_0"
			cat "Output_maxime/Task_order_effective_1" >> "Output_maxime/Task_order_effective_0"
			cat "Output_maxime/Task_order_effective_2" >> "Output_maxime/Task_order_effective_0"
		fi
	fi
fi
# Tracage des GFlops
gcc -o get_difference_between_orders get_difference_between_orders.c
./get_difference_between_orders Output_maxime/Task_order_HFP_0 Output_maxime/Task_order_effective_0 ${PATH_R}/R/Data/${DOSSIER}/Difference_between_orders/${FICHIER:0}.txt
Rscript ${PATH_R}/R/ScriptR/Difference_between_orders/Diff_HFP_HEFT_BW350_CM500.R ${PATH_R}/R/Data/${DOSSIER}/Difference_between_orders/${FICHIER}.txt $((GPU)) $((NT))
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Difference_between_orders/${FICHIER:0}.pdf
