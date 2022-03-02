#!/bin/bash
#	bash Scripts_maxime/DARTS/Draw_DARTS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Cholesky_dependances DARTS gemini-1-fgcs 1

PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
GPU=$6
NGPU=$7

#~ # Plot python GF sans légende
#~ python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv 0
#~ mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU_sanslegende.pdf
#~ # Plot python DT sans légende
#~ python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.csv 0
#~ mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU_sanslegende.pdf

# Plot python GF avec légende
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv 1
mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
# Plot python DT avec légende
python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.csv 1
mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

