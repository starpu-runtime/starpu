#!/usr/bin/bash
# ./cut_gflops_raw_out NOMBRE_DE_TAILLES_DE_MATRICES NOMBRE_ALGO_TESTE ECHELLE_X START_X

#~ cd..
#~ ./configure --enable-simgrid --disable-mpi --with-simgrid-dir=/home/gonthier/simgrid
#~ make install
start=`date +%s`
sudo make -j4
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
truncate -s 0 Output_maxime/GFlops_raw_out.txt
ulimit -S -s 50000
NB_ALGO_TESTE=6
NB_TAILLE_TESTE=10
ECHELLE_X=50
START_X=0
FICHIER=GF_M_M3D_N=15_BW350
echo "########## Random ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=random_order STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1  | tail -n 1 >> Output_maxime/GFlops_raw_out.txt
done
echo "########## Dmdar ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=dmdar STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1  | tail -n 1 >> Output_maxime/GFlops_raw_out.txt
done
echo "########## HFP ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 ORDER_U=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1  | tail -n 1 >> Output_maxime/GFlops_raw_out.txt
done
echo "########## HFP U ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1  | tail -n 1 >> Output_maxime/GFlops_raw_out.txt
done
echo "########## MST ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=mst STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1  | tail -n 1 >> Output_maxime/GFlops_raw_out.txt
done
echo "########## CM ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=cuthillmckee STARPU_NTASKS_THRESHOLD=30 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*10)) -z $((960*10)) -nblocks 10 -iter 1  | tail -n 1 >> Output_maxime/GFlops_raw_out.txt
done
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X
mv Output_maxime/GFlops_data_out.txt /home/gonthier/these_gonthier_maxime/Starpu/R/Data/Matrice3D/${FICHIER:0}.txt
Rscript /home/gonthier/these_gonthier_maxime/Starpu/R/ScriptR/Matrice3D/${FICHIER:0}.R
mv /home/gonthier/starpu/Rplots.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/Matrice3D/${FICHIER:0}.pdf
echo Fin du script
end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60)) "minutes."
