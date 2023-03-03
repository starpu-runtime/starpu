# bash Scripts_maxime/PlaFRIM-Grid5k/all_cholesky_dependances.sh

# oarsub -t exotic -p "network_address in ('gemini-1.lyon.grid5000.fr')" -r '2023-03-02 00:55:00' -l walltime=08:00:00 "bash Scripts_maxime/PlaFRIM-Grid5k/all_cholesky_dependances.sh"

# Opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 1920 12 2000 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 1920 12 2000 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 1920 12 2000 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 1920 12 2000 opti

bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 2880 12 4500 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 2880 12 4500 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 2880 12 4500 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 2880 12 4500 opti

bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 3840 12 8000 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 3840 12 8000 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 3840 12 8000 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 3840 12 8000 opti

# Opti sans limit de mem
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 1920 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 1920 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 1920 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 1920 12 0 opti

bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 2880 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 2880 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 2880 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 2880 12 0 opti

bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 1 3840 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 2 3840 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 4 3840 12 0 opti
bash Scripts_maxime/PlaFRIM-Grid5k/cholesky_dependances.sh 8 3840 12 0 opti
