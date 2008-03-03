##########################
# JASPERLIB (for grib 2) #
##########################


cd $JASPERSRC

wget -c "http://www.mmm.ucar.edu/wrf/src/wps_files/jasper-1.701.0.tar.gz"
tar -xvzf  jasper-1.701.0.tar.gz
cd jasper-1.701.0
./configure FC=gfortran-4.2 --prefix=$JASPERTARGET

make FC=gfortran-4.2
make install
