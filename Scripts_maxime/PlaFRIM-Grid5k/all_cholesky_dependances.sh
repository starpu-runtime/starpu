# bash Scripts_maxime/PlaFRIM-Grid5k/all_cholesky_dependances.sh

# oarsub -t exotic -p "network_address in ('gemini-1.lyon.grid5000.fr')" -r '2023-03-02 00:55:00' -l walltime=08:00:00 "bash Scripts_maxime/PlaFRIM-Grid5k/all_cholesky_dependances.sh"

# bash script_initialisation_starpu_maxime_sans_simgrid_grid5k

# no_prio
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 1920 12 2000 no_prio
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 1920 12 2000 no_prio
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 1920 12 2000 no_prio

# Best ones
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 1920 12 2000 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 1920 12 2000 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 1920 12 2000 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 1920 15 2000 best_ones

# Sans limit de mem
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 1920 12 0 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 1920 12 0 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 1920 12 0 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 1920 15 0 best_ones

# A enlever
# Best ones LU
#bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 1 1920 7 2000 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 2 1920 7 2000 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 4 1920 7 2000 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 8 1920 10 2000 best_ones

# Best ones sans limit de mem LU
bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 1 1920 10 0 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 2 1920 7 0 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 4 1920 7 0 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 8 1920 7 0 best_ones





#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 2880 12 4500 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 2880 12 4500 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 2880 12 4500 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 2880 12 4500 best_ones

#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 3840 12 8000 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 3840 12 8000 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 3840 12 8000 best_ones
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 3840 12 8000 best_ones

# Opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 1920 12 2000 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 1920 12 2000 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 1920 12 2000 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 1920 12 2000 opti

#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 2880 12 4500 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 2880 12 4500 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 2880 12 4500 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 2880 12 4500 opti

#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 3840 12 8000 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 3840 12 8000 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 3840 12 8000 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 3840 12 8000 opti

# Opti sans limit de mem
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 1920 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 1920 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 1920 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 1920 12 0 opti

#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 2880 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 2880 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 2880 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 2880 12 0 opti

#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 3840 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 3840 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 3840 12 0 opti
#bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 3840 12 0 opti
