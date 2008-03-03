##########################
# JASPERLIB (for grib 2) #
##########################


cd $ZLIBSRC

wget -c "http://www.zlib.net/zlib-1.2.3.tar.gz"
tar -xvzf  zlib-1.2.3.tar.gz
cd zlib-1.2.3 
./configure --prefix=$ZLIBTARGET

make
make install
