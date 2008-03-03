########
# JPEG #
########


cd $JPEGSRC

wget -c "http://www.ijg.org/files/jpegsrc.v6b.tar.gz"
tar -xvzf  jpegsrc.v6b.tar.gz
cd jpeg-6b
./configure --prefix=$JPEGTARGET

make
make install
