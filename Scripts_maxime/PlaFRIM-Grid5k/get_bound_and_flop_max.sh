# bash Scripts_maxime/PlaFRIM-Grid5k/get_bound_and_flop_max.sh

make -j 6
START_X=0
ECHELLE_X=5
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
truncate -s 0 ${FICHIER_RAW}

NB_TAILLE_TESTE=12

for ((i1=1 ; i1<=3; i1++))
do
	if [ $((i1)) == 1 ]; then TAILLE_TUILE=1920
	elif [ $((i1)) == 2 ]; then TAILLE_TUILE=2880
	elif [ $((i1)) == 3 ]; then TAILLE_TUILE=3840
	fi
	for ((i2=1 ; i2<=4; i2++))
	do
		if [ $((i2)) == 1 ]; then NGPU=1
		elif [ $((i2)) == 2 ]; then NGPU=2
		elif [ $((i2)) == 3 ]; then NGPU=4
		elif [ $((i2)) == 4 ]; then NGPU=8
		fi
		for ((i3=1 ; i3<=(($NB_TAILLE_TESTE)); i3++))
		do
			N=$((START_X+i3*ECHELLE_X))
			echo ${TAILLE_TUILE} ${NGPU} ${N}
			PRIORITY_ATTRIBUTION=1 STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((10)) STARPU_CUDA_PIPELINE=$((5)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((TAILLE_TUILE*N)) -nblocks $((N)) -bound | tail -n 4 >> ${FICHIER_RAW}
		done
	done
done

mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/Cholesky_dependances/Bound_and_flop_max.txt
