start=`date +%s`
make -j 6

echo "############################## 1 ##############################" 
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware gemini-2-ipdps 1
echo "############################## 2 ##############################" 
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware gemini-2-ipdps 2
echo "############################## 3 ##############################" 
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 1
echo "############################## 4 ##############################" 
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 2
echo "############################## 5 ##############################" 
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 3
echo "############################## 6 ##############################" 
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 4
#~ echo "############################## 7 ##############################" 
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 8

end=`date +%s`
runtime=$((end-start))
echo "Fin du script all_dynamic_data_aware.sh, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
