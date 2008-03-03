##########################
# JASPERLIB (for grib 2) #
##########################


cd $GRADSSRC

wget -c "ftp://iges.org/grads/2.0/grads-src-2.0.a0.tar.gz"
tar -xvzf  grads-src-2.0.a0.tar.gz
cd grads-2.0.a0

./configure --prefix=$GRADSTARGET SUPPLIBS=$GRADSTARGET --with-grib2  --with-nc

make
make install
