#!/usr/bin/bash
#bash Scripts_maxime/PlaFRIM/Script_pour_srun.sh

bash script_initialisation_starpu_maxime_sans_simgrid
module load linalg/mkl
bash Scripts_maxime/PlaFRIM/GF_Workingset.sh 9 Matrice_ligne
