#~ echo "Mémoire infinie"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_1GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_2GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 3
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_3GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 4
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_4GPU.txt

echo "Matrice 3D 1 GPU"
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice3D dynamic_data_aware_no_hfp 1
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_1GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_1GPU.txt
echo "Matrice 3D 2 GPU"
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 8 Matrice3D dynamic_data_aware_no_hfp 2
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_2GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_2GPU.txt

echo "Cholesky 1 GPU"
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Cholesky dynamic_data_aware_no_hfp 1
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_1GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_CHO_1GPU.txt
echo "Cholesky 2 GPU"
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Cholesky dynamic_data_aware_no_hfp 2
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_2GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_CHO_2GPU.txt

echo "Matrice 2D Sparse 1 GPU NO MEM LIMIT"
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_sparse_matrix 1
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_SPARSE_1GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M2D_SPARSE_1GPU.txt
#~ echo "Matrice 2D Sparse 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 2 Matrice_ligne dynamic_data_aware_no_hfp_sparse_matrix 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_SPARSE_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M2D_SPARSE_2GPU.txt

echo "Matrice 3D Sparse 1 GPU NO MEM LIMIT"
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Matrice3D dynamic_data_aware_no_hfp_sparse_matrix 1
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_SPARSE_1GPU.txt
mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_SPARSE_1GPU.txt
#~ echo "Matrice 3D Sparse 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 1 Matrice3D dynamic_data_aware_no_hfp_sparse_matrix 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_SPARSE_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_SPARSE_2GPU.txt