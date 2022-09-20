# Réserver sur Grid5k: oarsub -t exotic -p "network_address in ('gemini-1.lyon.grid5000.fr')" -r "2022-09-12 19:00:00" -l walltime=14:00:00
# Pour lancer en bash: oarsub -t exotic -p "network_address in ('gemini-1.lyon.grid5000.fr')" -l walltime=14:00:00 -r '2022-09-14 19:00:00' "bash Scripts_maxime/FGCS2021/Experiments.sh mgonthier"
# A lancer dans le dossier starpu/ avec bash Scripts_maxime/FGCS2021/Experiments.sh name_grid5k
NAME=$1
./configure --prefix=/home/${NAME}/starpu
make -j 100
make install

# Dans l'article
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 18 Matrice_ligne HFP 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M2D.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M2D.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice_ligne HFP_memory 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_MEMORY_M2D.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice3DZN HFP 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M3DZN.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M3DZN.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Cholesky HFP 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_CHO.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_CHO.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Random_task_order HFP 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M2D_RANDOM_ORDER.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M2D_RANDOM_ORDER.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Random_tasks HFP 1			
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M2D_RANDOM_TASKS.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M2D_RANDOM_TASKS.txt

# Pour la révision
#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 18 Sparse HFP 1 # Sparse 10%
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 30 Sparse HFP 1 # Sparse 10%
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_SPARSE.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_SPARSE.txt

#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Sparse HFP_N15 1 # Sparse de 10 à 100%
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_N15_SPARSE.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_N15_SPARSE.txt

#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Sparse HFP_N40 1 # Sparse de 10 à 100%
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_N40_SPARSE.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_N40_SPARSE.txt

#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Sparse3DZN HFP 1 # Sparse 10%
bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 20 Sparse3DZN HFP 1 # Sparse 10%
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_SPARSE_3DZN.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_SPARSE_3DZN.txt

#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Sparse3DZN HFP_N6 1 # Sparse de 10 à 100%
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_N6_SPARSE_3DZN.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_N6_SPARSE_3DZN.txt

#~ bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Sparse3DZN HFP_N14 1 # Sparse de 10 à 100%
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_N14_SPARSE_3DZN.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_N14_SPARSE_3DZN.txt






# Pas dans l'article
# bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice3D HFP 1
# mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_M3D.txt
# mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_M3D.txt
# bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 8 Sparse HFP_mem_infinie 1
# mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_SPARSE_INFINIE.txt
# mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_SPARSE_INFINIE.txt

# Pas dans l'article mais testé
# bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 8 Sparse HFP 1 2				
# mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_HFP_SPARSE.txt
# mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_HFP_SPARSE.txt
