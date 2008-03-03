#!/bin/bash

source ./meteo-env.sh

###########
# SCRIPTS #
###########

echo "Retrieve data ..."
cd $KATRINASCRIPTSDIR
./fetch-katrina-data.sh

###########
#
# For the single domain simulation 
#
###########

# cp $KATSINGLEDIR/namelist.wps $WPSSRC/WPS/namelist.wps
# cp $KATSINGLEDIR/namelist.input $WRFSRC/WRFV2/test/em_real/namelist.input 

###########
#
# For nested domain simulation 
#
###########

cp $KATNESTDIR/namelist.wps $WPSSRC/WPS/namelist.wps
cp $KATNESTDIR/namelist.input $WRFSRC/WRFV2/test/em_real/namelist.input 

#
# Pre-processing 
#

cd "$WPSSRC/WPS"
./link_grib.csh $KATRINAINPUT/Katrina/avn_* .

ln -sf "$WPSSRC/WPS/ungrib/Variable_Tables/Vtable.GFS" "$WPSSRC/WPS/Vtable"

./geogrid.exe
./ungrib.exe
./metgrid.exe

#
# Actual WRF processing 
#

cd "$WRFSRC/WRFV2/test/em_real"
ln -sf $WPSSRC/WPS/met_em* .
