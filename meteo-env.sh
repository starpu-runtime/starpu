#!/bin/bash

export WRF_NMM_NEST=1

export METEODIR=$PWD

export LOCALDIR=$METEODIR/LOCAL
export SRCDIR=$METEODIR/SRC

export NETCDFDIR=$SRCDIR/NETCDF/
export NETCDFSRC=$NETCDFDIR/
export NETCDFTARGET=$LOCALDIR
export NETCDF=$NETCDFTARGET

mkdir -p $LOCALDIR

# create the directories
mkdir -p $NETCDFDIR
mkdir -p $NETCDFSRC
mkdir -p $NETCDFTARGET

export MPICHDIR=$SRCDIR/MPICH/
export MPICHSRC=/home/gonnet/MPICH2-nmad/src
export MPICHTARGET=/home/gonnet/MPICH2-nmad/install/
#export MPICHSRC=$MPICHDIR
#export MPICHTARGET=$LOCALDIR

# create directories
mkdir -p $MPICHDIR
mkdir -p $MPICHSRC
mkdir -p $MPICHTARGET

export WRFDIR=$SRCDIR/WRF/
export WRFSRC=$WRFDIR
export WRFTARGET=$LOCALDIR

# create directories
mkdir -p $WRFDIR
mkdir -p $WRFSRC
mkdir -p $WRFTARGET

export JASPERDIR=$SRCDIR/JASPER/
export JASPERSRC=$JASPERDIR/
export JASPERTARGET=$LOCALDIR

mkdir -p $JASPERDIR
mkdir -p $JASPERSRC
mkdir -p $JASPERTARGET

export READLINEDIR=$SRCDIR/READLINE/
export READLINESRC=$READLINEDIR/
export READLINETARGET=$LOCALDIR

mkdir -p $READLINEDIR
mkdir -p $READLINESRC
mkdir -p $READLINETARGET

export ZLIBDIR=$SRCDIR/ZLIB/
export ZLIBSRC=$ZLIBDIR/
export ZLIBTARGET=$LOCALDIR

mkdir -p $ZLIBDIR
mkdir -p $ZLIBSRC
mkdir -p $ZLIBTARGET

export JPEGDIR=$SRCDIR/JPEG/
export JPEGSRC=$JPEGDIR/
export JPEGTARGET=$LOCALDIR

mkdir -p $JPEGDIR
mkdir -p $JPEGSRC
mkdir -p $JPEGTARGET


export PNGDIR=$SRCDIR/PNG/
export PNGSRC=$PNGDIR/
export PNGTARGET=$LOCALDIR

mkdir -p $PNGDIR
mkdir -p $PNGSRC
mkdir -p $PNGTARGET

export GDDIR=$SRCDIR/GD/
export GDSRC=$GDDIR/
export GDTARGET=$LOCALDIR

mkdir -p $GDDIR
mkdir -p $GDSRC
mkdir -p $GDTARGET


export G2CDIR=$SRCDIR/G2C/
export G2CSRC=$G2CDIR/
export G2CTARGET=$LOCALDIR

mkdir -p $G2CDIR
mkdir -p $G2CSRC
mkdir -p $G2CTARGET


export GRADSDIR=$SRCDIR/GRADS/
export GRADSSRC=$GRADSDIR/
export GRADSTARGET=$LOCALDIR

mkdir -p $GRADSDIR
mkdir -p $GRADSSRC
mkdir -p $GRADSTARGET


export UDUNITSDIR=$SRCDIR/UDUNITS/
export UDUNITSSRC=$UDUNITSDIR/
export UDUNITSTARGET=$LOCALDIR

mkdir -p $UDUNITSDIR
mkdir -p $UDUNITSSRC
mkdir -p $UDUNITSTARGET

export WPSDIR=$SRCDIR/WPS/
export WPSSRC=$WPSDIR/
export WPSTARGET=$LOCALDIR

mkdir -p $WPSDIR
mkdir -p $WPSSRC
mkdir -p $WPSTARGET

export SCRIPTSDIR=$METEODIR/SCRIPTS/
export POOLDIR=$METEODIR/POOL/
export METEOINPUT=$POOLDIR/METEOINPUT/
export KATRINAINPUT=$METEOINPUT/KATRINA

mkdir -p $POOLDIR
mkdir -p $METEOINPUT
mkdir -p $KATRINAINPUT

# to put the installation logs ...
export LOGDIR=$METEODIR/LOG
mkdir -p $LOGDIR

# there are various tests : either the current meteo, or katrina test suite
# either in single domain mode or nested mode.
export BXSCRIPTSDIR=$SCRIPTSDIR/BORDEAUX
export KATRINASCRIPTSDIR=$SCRIPTSDIR/KATRINA
export KATSINGLEDIR=$KATRINASCRIPTSDIR/SINGLE
export KATNESTDIR=$KATRINASCRIPTSDIR/NESTING
mkdir -p $BXSCRIPTSDIR
mkdir -p $KATRINASCRIPTSDIR
mkdir -p $KATSINGLEDIR
mkdir -p $KATNESTDIR

# the scripts to install the various dependencies 
export INSTALLSCRIPTSDIR=$SCRIPTSDIR/INSTALL
mkdir -p $INSTALLSCRIPTSDIR

# the directory containing the GEOGRAPHICAL data
export GEOGDATADIR=/mnt/scratch/meteo/geog
