echo "Mutex"

echo "Matrice 2D"
bash Scripts_maxime/PlaFRIM-Grid5k/Mutex_DARTS.sh 12 Matrice_ligne mutex_darts 4
mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_refined_mutex_M2D_4GPU.txt
mv Output_maxime/Data/Nb_conflit_donnee.txt Output_maxime/Nb_conflit_donnee_refined_mutex_M2D_4GPU.txt
mv Output_maxime/Data/Nb_conflit_donnee_critique.txt Output_maxime/Nb_conflit_donnee_critique_refined_mutex_M2D_4GPU.txt

#~ bash Scripts_maxime/PlaFRIM-Grid5k/Mutex_DARTS.sh 8 Matrice_ligne mutex_darts 4
#~ mv Output_maxime/GFlops_raw_out_1.txt Output_maxime/GF_linear_mutex_M2D_4GPU.txt
#~ mv Output_maxime/Data/Nb_conflit_donnee.txt Output_maxime/Nb_conflit_donnee_linear_mutex_M2D_4GPU.txt
