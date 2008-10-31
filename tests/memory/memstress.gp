#!/usr/bin/gnuplot -persist

set term postscript landscape color 22 
set output "memstress.ps"
set xlabel "Memory Pressure (MB)"
set ylabel "execution time degradation (%)"
set grid y
set key left top box
set datafile missing 'x'
plot "timings/memstress.data" usi 1:(( 100*(($2 / 2130) - 1))) with linespoint title "matrix size : 4096"  ,\
     "timings/memstress.data" usi 1:(( 100*(($3 / 16420) - 1) )) with linespoint title "8192"
