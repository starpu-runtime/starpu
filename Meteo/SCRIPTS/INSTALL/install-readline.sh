##########################
# JASPERLIB (for grib 2) #
##########################


cd $READLINESRC

wget -c "ftp://ftp.cwru.edu/pub/bash/readline-5.2.tar.gz"
tar -xvzf  readline-5.2.tar.gz
cd readline-5.2 
./configure --prefix=$READLINETARGET

make
make install
