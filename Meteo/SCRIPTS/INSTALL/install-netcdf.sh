#!/bin/bash

##########
# NETCDF #
##########

# get NetCDF sources
cd $NETCDFSRC
wget -c http://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf.tar.gz
tar -xvvzf netcdf.tar.gz > /dev/null
cd "./netcdf-3.6.2"

# compile NETCDF
./configure FC=gfortran-4.2 CPPFLAGS=-DgFortran --prefix=$NETCDFTARGET
make FC=gfortran-4.2
#make check

# install NETCDF
make install

