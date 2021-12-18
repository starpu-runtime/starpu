#~ echo "Mémoire infinie"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_1GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_2GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 3
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_3GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 4
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_4GPU.txt

#~ echo "Matrice 3D 1 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice3D dynamic_data_aware_no_hfp 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_1GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_1GPU.txt
#~ echo "Matrice 3D 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 8 Matrice3D dynamic_data_aware_no_hfp 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_2GPU.txt

#~ echo "Cholesky 1 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Cholesky dynamic_data_aware_no_hfp 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_1GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_CHO_1GPU.txt
#~ echo "Cholesky 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 7 Cholesky dynamic_data_aware_no_hfp 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_CHO_2GPU.txt

#~ echo "Matrice 2D Sparse 1 GPU NO MEM LIMIT"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice_ligne dynamic_data_aware_no_hfp_sparse_matrix 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_SPARSE_1GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M2D_SPARSE_1GPU.txt
#~ echo "Matrice 2D Sparse 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 5 Matrice_ligne dynamic_data_aware_no_hfp_sparse_matrix 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_SPARSE_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M2D_SPARSE_2GPU.txt

#~ echo "Matrice 3D Sparse 1 GPU NO MEM LIMIT"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 15 Matrice3D dynamic_data_aware_no_hfp_sparse_matrix 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_SPARSE_1GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_SPARSE_1GPU.txt
#~ echo "Matrice 3D Sparse 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 1 Matrice3D dynamic_data_aware_no_hfp_sparse_matrix 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_SPARSE_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_SPARSE_2GPU.txt

echo "Les variantes de Rebuttal_test.sh"

#~ echo "Matrice 3D 1 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 10 Matrice3D 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_1GPU_test.txt
#~ echo "Matrice 3D 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 8 Matrice3D 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_2GPU_test.txt
#~ echo "Matrice 3D 4 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 5 Matrice3D 4
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_4GPU_test.txt

#~ echo "Cholesky 1 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 10 Cholesky 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_1GPU_test.txt
#~ echo "Cholesky 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 7 Cholesky 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_2GPU_test.txt
echo "Cholesky 4 GPU" #Ici on fais 7 car on fais juste 2*5 pour l'echelle, sinon ca crash pour tout le monde
bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 7 Cholesky 4
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_4GPU_test.txt

#~ echo "Sparse 1 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 10 Sparse 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_SPARSE_1GPU_test.txt
#~ echo "Sparse 2 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 7 Sparse 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_SPARSE_2GPU_test.txt
#~ echo "Sparse 4 GPU"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 4 Sparse 4
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_SPARSE_4GPU_test.txt

#~ echo "Sparse 1 GPU infinite"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 10 Sparse_mem_infinite 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_SPARSE_INFINIE_1GPU_test.txt
#~ echo "Sparse 2 GPU infinite"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 7 Sparse_mem_infinite 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_SPARSE_INFINIE_2GPU_test.txt
#~ echo "Sparse 4 GPU infinite"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal_test.sh 4 Sparse_mem_infinite 4
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_SPARSE_INFINIE_4GPU_test.txt
