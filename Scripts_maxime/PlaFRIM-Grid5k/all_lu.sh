# bash Scripts_maxime/PlaFRIM-Grid5k/all_lu.sh

# oarsub -t exotic -p "network_address in ('gemini-1.lyon.grid5000.fr')" -r '2023-03-02 00:55:00' -l walltime=08:00:00 "bash Scripts_maxime/PlaFRIM-Grid5k/all_lu.sh"

# Best ones
bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 1 1920 7 2000 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 2 1920 7 2000 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 4 1920 7 2000 best_ones
bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 8 1920 7 2000 best_ones

# Best ones sans limit de mem
#~ bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 1 1920 7 0 best_ones
#~ bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 2 1920 7 0 best_ones
#~ bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 4 1920 7 0 best_ones
#~ bash Scripts_maxime/PlaFRIM-Grid5k/lu.sh 8 1920 7 0 best_ones
