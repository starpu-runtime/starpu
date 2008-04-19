#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "time.eps"
set title "Facto LU"
set logscale y
set xlabel "problem size (unknowns)"
set ylabel "execution time (ms)"
plot "time" usi 1:2 with lines title "1 cpu",\
     "time" usi 1:8 with lines title "4 cpus",\
     "time" usi 1:3 with lines title "1 cpu + 1 gpu",\
     "time" usi 1:7 with lines title "3 cpus + 1 gpu",\
     "time" usi 1:9 with lines title "4 cpus + 1 gpu"
