#!/bin/bash

source ./meteo-env.sh 

 ##########
 # NETCDF #
 ########## 
 
 echo "installing NETCDF ..."
 $INSTALLSCRIPTSDIR/install-netcdf.sh 1> $LOGDIR/log.out.netcdf 2> $LOGDIR/log.err.netcdf
 echo "done with NETCDF"
 
 ##########
 # MPICH2 #
 ##########
 
 echo "installing MPICH2 ..."
 $INSTALLSCRIPTSDIR/install-mpich2.sh 1> $LOGDIR/log.out.mpich2 2> $LOGDIR/log.err.mpich2
 echo "done with MPICH2"
  
  #########
  #  WRF  #
  #########
  
  echo "installing WRF ..."
  $INSTALLSCRIPTSDIR/install-wrf.sh 1> $LOGDIR/log.out.wrf 2> $LOGDIR/log.err.wrf
  echo "done with WRF"
   
  ##########################
  # JASPERLIB (for grib 2) #
  ##########################
  
  echo "installing JASPERLIB ..."
  $INSTALLSCRIPTSDIR/install-jasperlib.sh 1> $LOGDIR/log.out.jasperlib 2> $LOGDIR/log.err.jasperlib
  echo "done with JASPERLIB"
  
  ############
  # READLINE #
  ############
  
  echo "installing READLINE ..."
  $INSTALLSCRIPTSDIR/install-readline.sh 1> $LOGDIR/log.out.readline 2> $LOGDIR/log.err.readline
  echo "done with READLINE"
  
  ########
  # JPEG #
  ########
  
  echo "installing JPEG ..."
  $INSTALLSCRIPTSDIR/install-jpeg.sh 1> $LOGDIR/log.out.jpeg 2> $LOGDIR/log.err.jpeg
  echo "done with JPEG"
  
  ########
  # ZLIB #
  ########
  
  echo "installing ZLIB ..."
  $INSTALLSCRIPTSDIR/install-zlib.sh 1> $LOGDIR/log.out.zlib 2> $LOGDIR/log.err.zlib
  echo "done with ZLIB"
  
  ########
  # PNG #
  ########
  
  echo "installing PNG ..."
  $INSTALLSCRIPTSDIR/install-png.sh 1> $LOGDIR/log.out.png 2> $LOGDIR/log.err.png
  echo "done with PNG"
  
  ######
  # GD #
  ######
  
  echo "installing GD ..."
  $INSTALLSCRIPTSDIR/install-gd.sh 1> $LOGDIR/log.out.gd 2> $LOGDIR/log.err.gd
  echo "done with GD"
  
  #######
  # G2C #
  #######
  
  echo "installing G2C ..."
  $INSTALLSCRIPTSDIR/install-g2c.sh 1> $LOGDIR/log.out.g2c 2> $LOGDIR/log.err.g2c
  echo "done with G2C"
  
  ###########
  # UDUNITS #
  ###########
  
  echo "installing UDUNITS ..."
  $INSTALLSCRIPTSDIR/install-udunits.sh # 1> $LOGDIR/log.out.udunits 2> $LOGDIR/log.err.udunits
  echo "done with UDUNITS"
   
   
  #######
  # GRADS #
  #######
  
  echo "installing GRADS ..."
  $INSTALLSCRIPTSDIR/install-grads.sh 1> $LOGDIR/log.out.grads 2> $LOGDIR/log.err.grads
  echo "done with GRADS"
   
  
  
  #######
  # WPS #
  #######
  
  echo "installing WPS ..."
  $INSTALLSCRIPTSDIR/install-wps.sh 1> $LOGDIR/log.out.wps 2> $LOGDIR/log.err.wps
  echo "done with WPS"
