NAME=$1
PATH_STARPU=$2
PATH_R=$3

bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 18 Matrice_ligne HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Matrice_ligne HFP_memory 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Matrice3DZN HFP 1 9 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Cholesky HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 16 Random_task_order HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 16 Random_tasks HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 18 Sparse HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
