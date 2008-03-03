##########################
# JASPERLIB (for grib 2) #
##########################


cd $PNGSRC

wget -c "ftp://grads.iges.org/grads/Supplibs/2.0/src/libpng-1.2.18.tar.gz"
tar -xvzf libpng-1.2.18.tar.gz
cd libpng-1.2.18 
./configure --prefix=$PNGTARGET

make
make install
