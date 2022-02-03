# Lancer avec bash Scripts_maxime/Draw_ipdps2021/Draw.sh name_on_grid5k path_starpu path_r
NAME=$1
PATH_STARPU=$2
PATH_R=$3

bash Scripts_maxime/PlaFRIM-Grid5k/Draw_ipdps.sh 2 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 6 1 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_ipdps.sh 2 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 6 2 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_ipdps.sh 2 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 7 4 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_ipdps.sh 2 Random_task_order dynamic_data_aware_no_hfp_ipdps 6 2 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_ipdps.sh 2 Cholesky dynamic_data_aware_no_hfp 7 4 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_ipdps.sh 2 Sparse dynamic_data_aware_no_hfp 6 4 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_ipdps.sh 2 Sparse_mem_infinite dynamic_data_aware_no_hfp 7 4 ${NAME} ${PATH_STARPU} ${PATH_R}
