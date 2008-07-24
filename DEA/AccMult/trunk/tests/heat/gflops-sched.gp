#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "gflops-sched.eps"
set title "LU Decomposition : scheduling strategies"
set grid y
set key box
set xlabel "problem size (unknowns)"
set ylabel "GFlop/s"
set logscale x
#set pointsize	2
set key right bottom
set datafile missing 'x'
plot "timings/gflops.merged.data" usi 1:(2*$1*$1*$1 / (3*$2* 1000000)) with linespoint lt 3 title "greedy"  ,\
     "timings/gflops.merged.data" usi 1:(2*$1*$1*$1 / (3*$4* 1000000)) with linespoint title "prio" 

set output "gflops-sched-gain.eps"
set title "LU Decomposition : scheduling strategies : gain"
set grid y
set key box
set xlabel "problem size (unknowns)"
set ylabel "Gain"
set logscale x
#set pointsize	2
set key right bottom
set datafile missing 'x'
plot "timings/gflops.merged.data" usi 1:(100*(($2 / $4)-1)) with linespoint lt 3 title "gain"
