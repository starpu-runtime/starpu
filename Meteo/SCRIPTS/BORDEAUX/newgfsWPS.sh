#!/bin/bash

cd "$WPSSRC/WPS"

./link_grib.csh "$METEOINPUT/previsionsGFS2/gfs.t??z.*grb2f*"
y=`cat $METEOINPUT/previsionsGFS2/gfs.date.y`
yy=`cat $METEOINPUT/previsionsGFS2/gfs.date.yy`
ye=`cat $METEOINPUT/previsionsGFS2/gfs.date.ye`
t=`cat $METEOINPUT/previsionsGFS2/gfs.time`

cp $BXSCRIPTSDIR/namelist.wps.generic $POOLDIR/namelist.wps
sed -i "s/start_YYYY-MM-DD_HH/"$yy"_"$t"/g" $POOLDIR/namelist.wps 
sed -i "s/end_YYYY-MM-DD_HH/"$ye"_"$t"/g" $POOLDIR/namelist.wps
set -i "s/__GEOGDATADIR__/$GEOGDATADIR/g" $POOLDIR/namelist.wps

cp $POOLDIR/namelist.wps namelist.wps
ln -sf "$WPSSRC/WPS/ungrib/Variable_Tables/Vtable.GFS" "$WPSSRC/WPS/Vtable"

./geogrid.exe
./ungrib.exe
./metgrid.exe


