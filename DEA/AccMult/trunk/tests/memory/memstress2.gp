#!/usr/bin/gnuplot -persist

set term postscript landscape color 22 
set output "memstress2.ps"
set xlabel "Problem size"
set ylabel "execution time"
set logscale x
set key left top
set datafile missing 'x'
plot "timings/memstress2.data" usi 1:2 with lines title "reference"  ,\
     "timings/memstress2.data" usi 1:3 with lines title "350 MB"
