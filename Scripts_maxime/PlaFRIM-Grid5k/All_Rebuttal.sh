#~ echo "Mémoire infinie"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 5 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1 1920
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_1GPU_1920.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 5 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1 3840
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_1GPU_3840.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 5 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1 5760
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_1GPU_5760.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 5 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit 1 7680
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M2D_mem_no_limit_1GPU_7680.txt
#~ echo "Matrice 3D"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice3D dynamic_data_aware_no_hfp 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_1GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_1GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Matrice3D dynamic_data_aware_no_hfp 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_M3D_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_M3D_2GPU.txt
#~ echo "Cholesky"
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Cholesky dynamic_data_aware_no_hfp 1
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_1GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_CHO_1GPU.txt
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Rebuttal.sh 10 Cholesky dynamic_data_aware_no_hfp 2
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_HFP_CHO_2GPU.txt
#~ mv Output_maxime/GFlops_raw_out_3.txt Output_maxime/DT_HFP_CHO_2GPU.txt
