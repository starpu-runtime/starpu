# Lancer avec bash Scripts_maxime/Draw_fgcs2021/Draw.sh name_on_grid5k path_starpu path_r
NAME=$1
PATH_STARPU=$2
PATH_R=$3

bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_Draw_fgcs 6 1 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_Draw_fgcs 5 2 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_Draw_fgcs 6 4 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 13 Random_task_order dynamic_data_aware_no_hfp_Draw_fgcs 5 2 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 14 Cholesky dynamic_data_aware_no_hfp 6 4 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 15 Sparse dynamic_data_aware_no_hfp 5 4 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 15 Sparse_mem_infinite dynamic_data_aware_no_hfp 6 4 ${NAME} ${PATH_STARPU} ${PATH_R}
