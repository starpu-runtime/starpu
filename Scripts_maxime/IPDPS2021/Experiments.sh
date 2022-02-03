# To execute in the starpu/ folder with bash Scripts_maxime/IPDPS2021/Experiments.sh name_grid5k
NAME=$1
./configure --prefix=/home/${NAME}/starpu
make -j 100
make install
bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 1
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_DARTS_M2D_1GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_DARTS_M2D_1GPU.txt
bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 2
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_DARTS_M2D_2GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/Data/DT_DARTS_M2D_2GPU.txt
bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 4
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_DARTS_M2D_4GPU.txt
bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 13 Random_task_order dynamic_data_aware_no_hfp_ipdps 2
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_DARTS_M2D_RANDOM_ORDER_2GPU.txt
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 14 Cholesky dynamic_data_aware_no_hfp 4
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_DARTS_CHO_4GPU.txt
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Sparse dynamic_data_aware_no_hfp 4
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_DARTS_SPARSE_4GPU.txt
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Sparse_mem_infinite dynamic_data_aware_no_hfp 4
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/Data/GF_DARTS_SPARSE_INFINIE_4GPU.txt
