#~ After an execution in PlaFRIM
#~ bash Scripts_maxime/PlaFRIM/Draw.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ Matrice_ligne Workingset_PlaFRIM

scp mgonthie@plafrim:/home/mgonthie/starpu/Output_maxime/GFlops.txt /home/gonthier/starpu/Output_maxime/

PATH_STARPU=$1
PATH_R=$2
DOSSIER=$3
MODEL=$4
FICHIER=${PATH_STARPU}/starpu/Output_maxime/GFlops.txt

# Tracage des GFlops
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM/${DOSSIER}/GF_${MODEL}.txt ${MODEL} ${DOSSIER}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM/${DOSSIER}/GF_${MODEL}.pdf

