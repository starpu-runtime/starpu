#!/bin/bash
cd ~/Meteo/WPS/WPS
rm previsionsGFS2/* -f
./newgfsWhen.sh
./newgfsGet.sh
./newgfsWPS.sh
./newgfsWRF.sh
./newgfsGrads.sh



