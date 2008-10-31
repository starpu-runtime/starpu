#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "speedup.eps"
set title "Matrix multiplication"
set xlabel "number of CPUs"
set ylabel "speedup"
plot 	"timings/speedup.1024" usi 1:1	 with lines title "ideal",\
	"timings/speedup.1024" usi 1:($3/$2) with lines title "1024",\
	"timings/speedup.2048" usi 1:($3/$2) with lines title "2048",\
	"timings/speedup.4096" usi 1:($3/$2) with lines title "4096",\
	"timings/speedup.8192" usi 1:($3/$2) with lines title "8192",\
	"timings/speedup.16384" usi 1:($3/$2) with lines title "16384"
