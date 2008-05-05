#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "gflops.eps"
set title "Facto LU : impact of granularity"
set xlabel "problem size (unknowns)"
set ylabel "GFlops"
set logscale x
set key right bottom
set datafile missing 'x'
plot "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$2* 1000000)) with lines title "tile 128"  ,\
     "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$3* 1000000)) with lines title "tile 256"  ,\
     "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$4* 1000000)) with lines title "tile 512"  ,\
     "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$5* 1000000)) with lines title "tile 1024" ,\
     "timings/gflops.data" usi 1:(2*$1*$1*$1 / (3*$6* 1000000)) with lines title "tile 2048"
