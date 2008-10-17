#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "sched.eps"
set title "Matrix multiplication : impact of the scheduling policy"
set xlabel "matrices size"
set ylabel "GFlops"
set logscale x
set key right bottom
set datafile missing 'x'
plot "timings/gflops.data.greedy" usi 1:(2*$1*$1*$1 / ($2* 1000000)) with lines title "greedy"  ,\
     "timings/gflops.data.dm" usi 1:(2*$1*$1*$1 / ($2* 1000000)) with lines title "dm"
