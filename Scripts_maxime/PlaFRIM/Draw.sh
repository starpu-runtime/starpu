#~ After an execution in PlaFRIM
#~ bash Scripts_maxime/PlaFRIM/Draw.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ Matrice_ligne Workingset_PlaFRIM Sirocco10 PlaFRIM
#~ bash Scripts_maxime/PlaFRIM/Draw.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ Matrice_ligne Workingset_PlaFRIM Sirocco09 Grid5000

PATH_STARPU=$1
PATH_R=$2
DOSSIER=$3
MODEL=$4
GPU=$5
PLATEFORME=$6
FICHIER=${PATH_STARPU}/starpu/Output_maxime/GFlops.txt

if [ $PLATEFORME = "PlaFRIM" ]
	then
	scp mgonthie@plafrim:/home/mgonthie/starpu/Output_maxime/GFlops.txt /home/gonthier/starpu/Output_maxime/
fi
if [ $PLATEFORME = "Grid5000" ]
	then
	scp mgonthier@access.grid5000.fr:/home/mgonthier/lyon/starpu/Output_maxime/GFlops.txt /home/gonthier/starpu/Output_maxime/
fi
mv /home/gonthier/starpu/Output_maxime/GFlops.txt ${PATH_R}/R/Data/PlaFRIM/${DOSSIER}/GF_${MODEL}_${GPU}.txt

# Tracage des GFlops
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM/${DOSSIER}/GF_${MODEL}_${GPU}.txt ${MODEL} ${DOSSIER}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM/${DOSSIER}/GF_${MODEL}_${GPU}.pdf
