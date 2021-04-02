#!/usr/bin/bash
start=`date +%s`
#~ ./configure --enable-simgrid --disable-mpi --with-simgrid-dir=/home/gonthier/simgrid
#~ sudo make install
sudo make -j4

#### Matrice 2D ####
	#### 1 GPU ####
		#~ bash Scripts_maxime/Matrice_ligne/GF_M_MC_NT=225_LRU_BW350.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
		#~ bash Scripts_maxime/Matrice_ligne/GF_NT_MC_LRU_BW350_CM500.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
		#~ bash Scripts_maxime/Matrice_ligne/HFP_READY_THRESHOLD.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 9
	#### Multi GPU ####
		#~ bash Scripts_maxime/Matrice_ligne/GF_NT_MC_LRU_BW1050_CM167_MULTIGPU.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
		#~ bash Scripts_maxime/Matrice_ligne/GF_NT_MC_LRU_BW1050_CM250_MULTIGPU.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 9
		#~ bash Scripts_maxime/Matrice_ligne/GF_NT_MC_LRU_BW1050_CM500_MULTIGPU.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
		#~ bash Scripts_maxime/Matrice_ligne/HFP_MULTIGPU.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
		#~ bash Scripts_maxime/Matrice_ligne/HFP_READY_THRESHOLD_3GPU.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 9
	#### Test ####
		#~ bash Scripts_maxime/Matrice_ligne/TEST_M2D_1GPU.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8
	#### Difference between orders ####
		#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Diff_HFP_HEFT_BW350_CM500_1GPU HEFT 1
		#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Diff_HFP_HEFT_BW350_CM500_3GPU HEFT 3
		#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Diff_HFP_HFPR_BW350_CM500_1GPU HFPR 1
	#### HMETIS ####
		#~ bash Scripts_maxime/Matrice_ligne/GF_NT_MC_LRU_BW1050_CM250_MULTIGPU_HMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 9
	#### Varying threshold ####
		#~ bash Scripts_maxime/Matrice_ligne/GF_TH_MC_LRU_BW1050_CM250_MULTIGPU.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
	
#### Matrice 3D ####
	#### 1 GPU ####
		#~ bash Scripts_maxime/Matrice3D/GF_M_M3D_N=15_BW350.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
		#~ bash Scripts_maxime/Matrice3D/GF_NT_M3D_BW350_CM500.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8
		#~ bash Scripts_maxime/Matrice3D/GF_NT_M3D_BW350_CM500_DMDARFIXED_MODULARHEFT.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8
	#### Multi GPU ####
		bash Scripts_maxime/Matrice3D/GF_NT_M3D_3GPU_BW1050_CM250.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8
	#### Difference between orders ####
		#~ bash Scripts_maxime/get_difference_between_orders.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice3D Diff_HFP_HEFT_BW350_CM500
	#### Varying threshold ####
		#~ bash Scripts_maxime/Matrice3D/GF_TH_M3D_3GPU_BW1050_CM250.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8

#### Cholesky ####
	#~ bash Scripts_maxime/Cholesky/GF_M_CHO_N=20_BW350.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 9
	#~ bash Scripts_maxime/Cholesky/GF_NT_CHO_BW350_CM500.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 7

#### Random tasks ####
	#~ bash Scripts_maxime/Random_tasks/GF_M_MC_LRU_N=15_BW350_CM500_RANDOMTASKS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
	#~ bash Scripts_maxime/Random_tasks/GF_NT_MC_LRU_BW350_CM500_RANDOMTASKS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10


#~ bash Scripts_maxime/CMvsRCM/GF_NT_CMvsRCM_MC.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10
#~ bash Scripts_maxime/CMvsRCM/GF_NT_CMvsRCM_CHO.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 7

#### hMETIS ####
	#~ bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne UBfactor
	#~ bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 20 Matrice_ligne Nruns
	#~ bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice_ligne CType
	#~ bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 3 Matrice_ligne RType
	#~ bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne Reconst
	#~ bash Scripts_maxime/hMETIS/hMETIS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 4 Matrice_ligne Vcycle

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
