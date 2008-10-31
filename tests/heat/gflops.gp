#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "gflops.eps"
#set title "Facto LU : impact of granularity"
set grid y
set key box
set xlabel "problem size (unknowns)"
set ylabel "GFlop/s"
set logscale x
#set pointsize	2
set key right bottom
set datafile missing 'x'
plot "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$2* 1000000)) with linespoint lt 3 title "block size : 128"  ,\
     "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$3* 1000000)) with linespoint title "block size : 256"  ,\
     "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$4* 1000000)) with linespoint title "block size : 512"  ,\
     "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$5* 1000000)) with linespoint title "block size : 1024"
