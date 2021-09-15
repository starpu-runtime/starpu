start=`date +%s`
make -j 6

bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 1
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 2
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 3
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 4
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 8
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware_no_hfp Attila 1
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware_no_hfp Attila 2
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware_no_hfp Attila 3
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware_no_hfp Attila 4
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware_no_hfp Attila 8

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
