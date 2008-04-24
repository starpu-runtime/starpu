#!/bin/bash

# generate the input data
./fxt-tool $1

# generate the gnuplot script 
echo "#!/usr/bin/gnuplot -persist" 			> histo.gp
echo "set term postscript eps enhanced color" 		>> histo.gp
echo "set output \"histo.eps\""				>> histo.gp
echo "set boxwidth 0.75 absolute" 			>> histo.gp
echo "set boxwidth 0.75 absolute"			>> histo.gp
echo "set nokey"			>> histo.gp
echo "set style fill   solid 1.00 border -1"		>> histo.gp
#echo "set key invert samplen 4 spacing 1 width 0 height 0"	>> histo.gp
#echo "set style histogram rowstacked title  offset character 0, 0, 0"	>> histo.gp
echo "set style histogram rowstacked"	>> histo.gp
echo "set style data histograms"			>> histo.gp
echo "set xtics border in scale 1,0.5 nomirror rotate by -45  offset character 0, 0, 0"	>> histo.gp

#and the actual plotting part :
ncols=`head -1 data |wc -w`

color="red"

echo "plot \\"		>> histo.gp
for i in `seq 1 $(($ncols - 2))`
do

if [ $color == "red" ]; 
then
	color="green"
else
	color="red"
fi


echo "\"data\" using $i lt rgb \"$color\",\\"	>> histo.gp
done

if [ $color == "red" ]; 
then
	color="green"
else
	color="red"
fi


echo "\"data\" using $(($ncols - 1)) lt rgb \"$color\""	>> histo.gp

# now call the script
gnuplot histo.gp
