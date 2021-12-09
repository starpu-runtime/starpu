#~ bash Scripts_maxime/all_dynamic_data_aware.sh
start=`date +%s`
make -j 6

#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware gemini-2-ipdps 1
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware gemini-2-ipdps 2
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 1
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 2
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 3
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 4
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 8
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Random_task_order dynamic_data_aware_no_hfp Attila 1
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Random_task_order dynamic_data_aware_no_hfp Attila 2

#	For the rebuttal
echo "M3D"
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 7 Matrice3D dynamic_data_aware_no_hfp gemini-1-fgcs 1
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice3D dynamic_data_aware_no_hfp gemini-1-fgcs 2
echo "CHO"
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 7 Cholesky dynamic_data_aware_no_hfp gemini-1-fgcs 1
bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 4 Cholesky dynamic_data_aware_no_hfp gemini-1-fgcs 2
#~ bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit gemini-1-fgcs 1

end=`date +%s`
runtime=$((end-start))
echo "Fin du script all_dynamic_data_aware.sh, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
