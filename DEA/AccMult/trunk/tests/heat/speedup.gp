#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "speedup.eps"
set title "Facto LU : impact of granularity"
set xlabel "problem size (unknowns)"
set ylabel "speedup ( 3 Cores + 1 GPU vs . 4 Cores )"
set yrange [0.5:2.5]
plot "speedup.8" usi 1:($3/$2) with lines title "(8x8)",\
	"speedup.16" usi 1:($3/$2) with lines title "(16x16)",\
	"speedup.32" usi 1:($3/$2) with lines title "(32x32)",\
	"speedup.8" usi 1:((1))	with lines title "Reference"
