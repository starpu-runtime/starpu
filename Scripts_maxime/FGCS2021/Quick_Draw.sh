# Lancer avec bash Scripts_maxime/FGCS2021/Quick_Draw.sh name_on_grid5k path_starpu path_r
NAME=$1
PATH_STARPU=$2
PATH_R=$3

bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Matrice_ligne HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Matrice_ligne HFP_memory 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 6 Matrice3D HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 5 Matrice3D HFP 1 9 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 5 Cholesky HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Random_task_order HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Random_tasks HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
