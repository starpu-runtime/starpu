#!/bin/bash

outputfile=dmda.data

gnuplot > /dev/null << EOF

set term postscript eps enhanced color
set output "dmda.eps"

#set pointsize 0.75
set grid y
set grid x
#set logscale x
set xtics 4096,4096,30760
set size 0.65

set xlabel "Matrix size"

set xtics 4096,4096,30760

set size 0.8,0.8

set multiplot

set size 0.8,0.4
set origin 0.0,0.0
set ylabel "Cache miss ratio"
set format y "%.0f %%"
set key box right top title "Penality for non-local accesses"
plot "$outputfile" usi 1:2 with lines lt rgb "black"  lw 2 title "with" ,\
     "$outputfile" usi 1:3 with lines lt rgb "black"  lw 2 title "without"

set size 0.8,0.4
set origin 0.0,0.4
set ylabel "Total data transfers"
set format y "%.0f GB"
set key  box left top title "Penality for non-local accesses"
plot "$outputfile" usi 1:((\$4/1024)) with linespoint lt rgb "black" lw 2 title "with",\
     "$outputfile" usi 1:((\$5/1024)) with linespoint lt rgb "black"  lw 2 title "without"

unset multiplot

EOF

