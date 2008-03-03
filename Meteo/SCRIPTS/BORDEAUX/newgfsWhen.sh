#!/bin/bash
cd $METEOINPUT
#regions=( atlantique mediteranee )
#echo "number of regions : "${#regions[@]



yh=`date -d "1 days ago" +"%Y%m%d"`
yt=`date +"%Y%m%d"`


yhh=`date -d "1 days ago" +"%Y-%m-%d"`
ytt=`date +"%Y-%m-%d"`

yhe=`date +"%Y-%m-%d"`
yte=`date -d "1 days" +"%Y-%m-%d"`

y="0";

echo $yh;
echo $yt;

	

rm -f gfs_when

wget -c "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yh"00/gfs.t00z.pgrbf180.idx" -O gfs_when
size=`wc -c <gfs_when`;
ls -la gfs_when
if test "$size" -gt "10"; then
	t="00";
 	y=$yh;
 	yy=$yhh;
	ye=$yhe;
fi

#wget "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yh"06/gfs.t06z.pgrbf180.idx" -O gfs_when
#size=`wc -c <gfs_when`;
#ls -la gfs_when
#if test "$size" -gt "10"; then
#	t="06";
# 	y=$yh;
# 	yy=$yhh;
#	ye=$yhe;
#fi

#wget "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yh"12/gfs.t12z.pgrbf180.idx" -O gfs_when
#size=`wc -c <gfs_when`;
#ls -la gfs_when
#if test "$size" -gt "10"; then
#	t="12";
# 	y=$yh;
# 	yy=$yhh;
#	ye=$yhe;
#fi

#wget "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yh"18/gfs.t18z.pgrbf180.idx" -O gfs_when
#size=`wc -c <gfs_when`;
#ls -la gfs_when
#if test "$size" -gt "10"; then
#	t="18";
# 	y=$yh;
# 	yy=$yhh;
#	ye=$yhe;
#fi

wget -c "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yt"00/gfs.t00z.pgrbf180.idx" -O gfs_when
size=`wc -c <gfs_when`;
ls -la gfs_when
#if test "$size" -gt "10"; then
	t="00";
 	y=$yt;
 	yy=$ytt;
	ye=$yte;
#fi

#wget "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yt"06/gfs.t06z.pgrbf180.idx" -O gfs_when
#size=`wc -c <gfs_when`;
#ls -la gfs_when
#if test "$size" -gt "10"; then
#	t="06";
# 	y=$yt;
# 	yy=$ytt;
#	ye=$yte;
#fi

#wget "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yt"12/gfs.t12z.pgrbf180.idx" -O gfs_when
#size=`wc -c <gfs_when`;
#ls -la gfs_when
#if test "$size" -gt "10"; then
#	t="12";
# 	y=$yt;
# 	yy=$ytt;
#	ye=$yte;
#fi

#wget "http://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs."$yt"18/gfs.t18z.pgrbf180.idx" -O gfs_when
#size=`wc -c <gfs_when`;
#ls -la gfs_when
#if test "$size" -gt "10"; then
#	t="18";
# 	y=$yt;
# 	yy=$ytt;
#	ye=$yte;
#fi


echo "---------- 4			---------- ";
	
echo "Newest Dataset "$t;
echo "Newest Timeset "$y;

mkdir -p previsionsGFS2

if [ $y != "0" ];  then
	echo $t >previsionsGFS2/gfs.time;
	echo $y >previsionsGFS2/gfs.date.y;
	echo $yy >previsionsGFS2/gfs.date.yy;
	echo $ye >previsionsGFS2/gfs.date.ye;
fi




