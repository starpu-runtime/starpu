#!/bin/bash

source ./meteo-env.sh

###########
# SCRIPTS #
###########

echo "Get the date ... "
cd $BXSCRIPTSDIR
#./newgfsWhen.sh
./newgfsWhenNCEPmaster.sh

echo "Retrieve data ..."
cd $BXSCRIPTSDIR
#./newgfsGet.sh
./newgfsGetNCEPMaster.sh

echo "WPS ... "
cd $BXSCRIPTSDIR
./newgfsWPS.sh

echo "WRF ..."
cd $BXSCRIPTSDIR
./newgfsWRF.sh

#cd "$WRFSRC/WRFV2/test/em_real"
#$MPICHTARGET/bin/mpd -d
#$MPICHTARGET/bin/mpirun -np 8 ./real.exe
#$MPICHTARGET/bin/mpirun -np 8 ./wrf.exe
