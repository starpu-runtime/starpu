#!/bin/bash

#######
# WPS #
#######

# require libpng12-dev

cd $WPSSRC

wget -c "http://www.mmm.ucar.edu/wrf/src/WPSV2.2.1.TAR.gz"
tar -xvzf WPSV2.2.1.TAR.gz
cd WPS
echo '9' |  JASPERLIB=$JASPERTARGET/lib/ JASPERINC=$JASPERTARGET/include/ ./configure

sed -r "s@...WRFV2@$WRFSRC/WRFV2/@" configure.wps > configure.wps.tmp
sed -r "s@g95@gfortran-4.2@g" configure.wps.tmp > configure.wps
sed -r "s@gcc@gcc-4.2@g" configure.wps > configure.wps.tmp
sed -r "s@/data3a/mp/gill/WPS_LIBS/local/@$JASPERTARGET@g" configure.wps.tmp > configure.wps

rm configure.wps.tmp

./compile

