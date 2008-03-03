#!/bin/bash

#########
#  WRF  #
#########

# get the sources 
cd $WRFSRC
#wget -c "http://www.mmm.ucar.edu/wrf/src/WRFV2.2.1.TAR.gz"
wget -c "http://www.mmm.ucar.edu/wrf/src/WRFV2.2.TAR.gz"
tar -xvf "WRFV2.2.TAR.gz"

WRFPATCHES=$METEODIR/PATCHES/WRF

# patch the sources for gfortran !
cd WRFV2
patch -p1 < $WRFPATCHES/WRFV2-gfortran-x86_64-1.patch

# configure WRF
echo "14" | NETCDF=$NETCDFTARGET FC=gfortran-4.2 ./configure

# tune some parameters by hand ....
CMDMPIF90="$MPICHTARGET/bin/mpif90 -f90=gfortran-4.2"
CMDMPICC="$MPICHTARGET/bin/mpicc"
sed -r "s@mpif90@$CMDMPIF90@" configure.wrf > configure.wrf.tmp
sed -r "s@mpicc@$CMDMPICC@" configure.wrf.tmp > configure.wrf.tmp2
sed -r "s@gcc@gcc-4.2@" configure.wrf.tmp2 > configure.wrf.tmp
sed -r "s@\(^SCC.*\)gcc@/1gcc-4.2@" configure.wrf.tmp > configure.wrf.tmp2
sed -r "s@\(^SFC.*\)gfortran@/1gfortran-4.2@" configure.wrf.tmp2 > configure.wrf.tmp
sed -r "s@\(^CC_TOOLS.*\)gcc@/1gcc-4.2@" configure.wrf.tmp > configure.wrf.tmp2
rm configure.wrf.tmp
mv configure.wrf.tmp2 configure.wrf

# compile WRF
./compile wrf
./compile em_real

