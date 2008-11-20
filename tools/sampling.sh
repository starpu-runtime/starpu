#!/bin/bash

inputfile=$1

archlist=`cut -f 1 $inputfile | sort | uniq | xargs` 
hashlist=`cut -f 2 $inputfile | sort | uniq | xargs` 

# extract subfiles from the history file
for arch in $archlist
do
	for h in $hashlist
	do
		echo "pouet $arch - $h "
		grep "^$arch	$h" $inputfile > $inputfile.$arch.$h
	done
done

# create the gnuplot file

gpfile=$inputfile.gp

echo "#!/usr/bin/gnuplot -persist" 		> $gpfile
echo "set term postscript eps enhanced color" 	>> $gpfile
echo "set logscale x"				>> $gpfile 
echo "set logscale y"				>> $gpfile 
echo "set output \"$inputfile.eps\""		>> $gpfile

echo -n "plot	" 				>> $gpfile

first=1

for arch in $archlist
do
	for h in $hashlist
	do
		if [ $first = 0 ] 
		then
			echo -n "  , " >> $gpfile
		else
			first=0
		fi

		echo -n " \"$inputfile.$arch.$h\" using 3:4  title \"arch $arch hash $h\" " >> $gpfile
	done
done

gnuplot $gpfile
