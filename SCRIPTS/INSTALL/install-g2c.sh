#######
# G2C #
#######


cd $G2CSRC

wget -c "http://www.nco.ncep.noaa.gov/pmb/codes/GRIB2/g2clib-1.0.5.tar"
tar -xvf  g2clib-1.0.5.tar
cd g2clib-1.0.5

make ARFLAGS=" " CFLAGS+=" -I$G2CTARGET/include/"

cp grib2.h $G2CTARGET/include/
cp libgrib2c.a $G2CTARGET/lib/
