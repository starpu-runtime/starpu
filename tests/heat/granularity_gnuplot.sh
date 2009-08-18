#!/bin/bash

outputfile=granularity.data

find timing/* -name "granularity.*" > filelist

grainlist=`sed -e "s/.*granularity\.\(.*\)\.\(.*\)$/\1/" filelist |sort -n|uniq|xargs`
sizelist=`sed -e "s/.*granularity\.\(.*\)\.\(.*\)$/\2/" filelist |sort -n|uniq`

# Make some header 
line="#SIZE	"
for grain in $grainlist
do
	line="$line	$grain"
done
echo "$line" > $outputfile

for size in $sizelist
do
	line="$size	"

	for grain in $grainlist
	do
		# Compute Average value ...

		if test -f timing/granularity.$grain.$size; then
			# file does exists 
			filename=timing/granularity.$grain.$size

			# echo "GRAIN $grain SIZE $size exists !"
			# how many samples do we have ?
			nsample=`cat $filename | wc -w`
			if test $nsample -ge 1; then
				sum=0
				for i in `cat $filename | xargs`
				do
					sum=$(echo "$sum + $i"|bc -l)
				done
				
				# average execution time is ...
				mean=$(echo "$sum / $nsample"|bc -l)

				# in Flop/s this is 2*size^3/3
				gflops=$(echo "2.0 * $size * $size * $size / (3000000 * $mean)"|bc -l)

				# just make this a bit prettier ..
				gflops=`echo $gflops | sed -e "s/\(.*\.[0-9][0-9]\).*$/\1/"` 

				line="$line     $gflops"
			else
				# we have no valid sample even if the file exists
				line="$line     x"
			fi 
		else
			# file does not exist
			line="$line     x"
		fi
		
		line="$line	"
	done

	echo "$line" >> $outputfile
done

gnuplotline="plot "
cnt=2
for grain in $grainlist
do
	if test $cnt -ne 2; then
		# i hate gnuplot :)
		gnuplotline="$gnuplotline , "
	fi
	gnuplotline="$gnuplotline \"$outputfile\" usi 1:$cnt with linespoint title \"\($grain x $grain\)\" lt rgb \"black\" "
	cnt=$(($cnt+1))
done

gnuplot > /dev/null << EOF

set term postscript eps enhanced color
set output "granularity.eps"


set pointsize 0.75
#set title "Impact of granularity"
set grid y
set grid x
#set logscale x
set key box right bottom title "tile size"
set size 0.65

set xlabel "Matrix size"
set ylabel "GFlop/s"


$gnuplotline

EOF

