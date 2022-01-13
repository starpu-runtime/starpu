# A lancer dans le dossier starpu/ avec bash Scripts_maxime/FGCS2021/Experiments.sh name_grid5k
NAME=$1
./configure --prefix=/home/${NAME}/starpu
make -j 100
make install
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Matrice_ligne HFP 1 				V
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M2D.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M2D.txt
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice_ligne HFP_memory 1 		# X
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_MEMORY_M2D.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice3D HFP 1					V
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M3D.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M3D.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 9 Cholesky HFP 1						X mais deja commencé
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_CHO.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_CHO.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Random_task_order HFP 1			V
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M2D_RANDOM_ORDER.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M2D_RANDOM_ORDER.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Random_tasks HFP 1					V
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M2D_RANDOM_TASKS.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M2D_RANDOM_TASKS.txt
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 8 Sparse HFP 1
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_SPARSE.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_SPARSE.txt
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 8 Sparse HFP_mem_infinie 1
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_SPARSE_INFINIE.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_SPARSE_INFINIE.txt
