######
# GD #
######


cd $GDSRC

wget -c "http://www.libgd.org/releases/gd-2.0.35.tar.gz"
tar -xvzf gd-2.0.35.tar.gz
cd gd-2.0.35

aclocal
autoconf
./configure --with-jpeg --with-png --prefix=$GDTARGET

make
make install
