#!/bin/bash

y=`cat $METEOINPUT/previsionsGFS2/gfs.date.y`
yy=`cat $METEOINPUT/previsionsGFS2/gfs.date.yy`
ye=`cat $METEOINPUT/previsionsGFS2/gfs.date.ye`
t=`cat $METEOINPUT/previsionsGFS2/gfs.time`
cd "$WRFSRC/WRFV2/test/em_real"

cp $BXSCRIPTSDIR/namelist.input.generic $POOLDIR/namelist.input

sed -i "s/start_YYYY/"${yy:0:4}"/g" $POOLDIR/namelist.input
sed -i "s/start_MM/"${yy:5:2}"/g" $POOLDIR/namelist.input
sed -i "s/start_DD/"${yy:8:2}"/g" $POOLDIR/namelist.input
sed -i "s/start_HH/"$t"/g" $POOLDIR/namelist.input

sed -i "s/end_YYYY/"${ye:0:4}"/g" $POOLDIR/namelist.input
sed -i "s/end_MM/"${ye:5:2}"/g" $POOLDIR/namelist.input
sed -i "s/end_DD/"${ye:8:2}"/g" $POOLDIR/namelist.input
sed -i "s/end_HH/"$t"/g" $POOLDIR/namelist.input
cp $POOLDIR/namelist.input namelist.input

ln -sf $WPSSRC/WPS/met_em* .
# mpirun -machinefile machines -np 8 ./real.exe
# mpirun -machinefile machines -np 8 ./wrf.exe
