# Lancer avec bash Scripts_maxime/FGCS2021/Draw.sh name_on_grid5k path_starpu path_r
# Donc pour moi cela donne : bash Scripts_maxime/FGCS2021/Draw.sh mgonthier /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/
NAME=$1
PATH_STARPU=$2
PATH_R=$3

#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 16 Matrice_ligne HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Matrice_ligne HFP_memory 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 10 Matrice3D HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 9 Cholesky HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 16 Random_task_order HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 16 Random_tasks HFP 1 8 ${NAME} ${PATH_STARPU} ${PATH_R}
#~ bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 8 Sparse HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
bash Scripts_maxime/PlaFRIM-Grid5k/Draw_fgcs.sh 8 Sparse_mem_infinite HFP 1 10 ${NAME} ${PATH_STARPU} ${PATH_R}
