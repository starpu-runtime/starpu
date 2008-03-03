#!/bin/sh

source ./meteo-env.sh

# remove the meteo initial input 
rm -f $POOLDIR/namelist*
rm -rf $METEOINPUT

# remove WPS output and intermediate files
cd "$WPSSRC/WPS/"
rm -f *.nc FILE* *.log
for i in GRIBFILE.*
do
	unlink $i 2> /dev/null
done

# remove WRF outputs and intermediate files
cd "$WRFSRC/WRFV2/test/em_real"
rm -f wrfout_d* rsl.* wrfinput* wrfbdy*

for i in *.nc
do
	unlink $i 2> /dev/null
done
