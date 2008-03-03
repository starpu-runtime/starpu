##########################
# UDUNITS #
##########################

echo "LA"
cd $UDUNITSSRC
echo "ICI"

rm -f udunits-1.12.4.tar
wget -c "ftp://ftp.unidata.ucar.edu/pub/udunits/udunits-1.12.4.tar.Z"
echo "wget ok" 
uncompress udunits-1.12.4.tar.Z
echo "1"
tar -xvf udunits-1.12.4.tar
echo "2"
cd udunits-1.12.4/src 
echo "3"

aclocal
./configure --prefix=$UDUNITSTARGET
make CC=gcc CFLAGS+=" -Df2cFortran -fPIC " LDFLAGS+=" -lm " PERL=perl
make install CC=gcc CFLAGS+=" -Df2cFortran -fPIC " LDFLAGS+=" -lm " PERL=perl
make test
