#~ After an execution in PlaFRIM
#~ bash Scripts_maxime/PlaFRIM/Draw.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ Matrice_ligne Workingset_PlaFRIM Sirocco10 PlaFRIM
#~ bash Scripts_maxime/PlaFRIM/Draw.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ Matrice_ligne Workingset_PlaFRIM Sirocco09 Grid5000
#~ bash Scripts_maxime/PlaFRIM/Draw.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ Matrice_ligne Workingset_PlaFRIM Gemini02 Grid5000 mgonthie
#~ bash Scripts_maxime/PlaFRIM/Draw.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ Matrice_ligne Memory_PlaFRIM Gemini02 Grid5000

PATH_STARPU=$1
PATH_R=$2
DOSSIER=$3
MODEL=$4
GPU=$5
PLATEFORME=$6
NAME=$7
FICHIER=${PATH_STARPU}/starpu/Output_maxime/GFlops.txt

if [ $PLATEFORME = "PlaFRIM" ]
	then
	scp ${NAME}@plafrim:/home/${NAME}/starpu/Output_maxime/GFlops.txt ${PATH_STARPU}/starpu/Output_maxime/
fi
if [ $PLATEFORME = "Grid5000" ]
	then
	scp ${NAME}@access.grid5000.fr:/home/${NAME}/lyon/starpu/Output_maxime/GFlops.txt ${PATH_STARPU}/starpu/Output_maxime/
fi
mv ${PATH_STARPU}/starpu/Output_maxime/GFlops.txt ${PATH_R}/R/Data/PlaFRIM/${DOSSIER}/GF_${MODEL}_${GPU}.txt

# Tracage des GFlops
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/PlaFRIM/${DOSSIER}/GF_${MODEL}_${GPU}.txt ${MODEL} ${DOSSIER} ${GPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/PlaFRIM/${DOSSIER}/GF_${MODEL}_${GPU}.pdf
