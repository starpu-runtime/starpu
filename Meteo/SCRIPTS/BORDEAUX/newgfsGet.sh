#!/bin/bash
cd $METEOINPUT
y=`cat previsionsGFS2/gfs.date.y`
t=`cat previsionsGFS2/gfs.time`


one=0
while test "$one" -lt "25"
do
	if test "$one" -lt "10"; then
        	one="0"$one
        fi
	cd previsionsGFS2	
	wget -c "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$y$t"/gfs.t"$t"z.pgrb2f"$one -O "gfs.t"$t"z.pgrb2f"$one
	#wget -c http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$y$t"/gfs.t"$t"z.pgrbf"$one"
	#wget http://nomad3.ncep.noaa.gov/pub/gfs_master/gfs"$y"/gfs.t00z.master.grbf"$one";
	#wget http://nomads6.ncdc.noaa.gov/pub/raid1b/gfs_master/gfs"$y"/gfs.t00z.master.grbf"$one"
	cd .. 

	one=`expr $one + 6`
done

