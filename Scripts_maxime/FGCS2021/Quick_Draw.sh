# Lancer avec bash Scripts_maxime/FGCS2021/Quick_Draw.sh name path_starpu path_r
NAME=$1
PATH_STARPU=$2
PATH_R=$3

bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 2 Matrice_ligne HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 2 Matrice_ligne HFP_memory 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 2 Matrice3D HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 2 Cholesky HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 2 Random_task_order HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 2 Random_tasks HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
