#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color

set logscale x
set logscale y
set zrange [1:45000]
set hidden3d
set dgrid3d 128,32

set output "uncoalesced.eps"
set xlabel "nblocks"
set ylabel "nthreads"
set zlabel "bandwith (MB/s)"
splot "perf.log.2" using ($1*$2):($1*$2*$3*$4):6 with lines 

set output "coalesced.eps"
set xlabel "nblocks"
set ylabel "nthreads"
set zlabel "bandwith (MB/s)"
splot "perf.log.3" using ($1*$2):($1*$2*$3*$4):6 with lines 

set output "both.eps"
set xlabel "nblocks"
set ylabel "nthreads"
set zlabel "bandwith (MB/s)"
splot "perf.log.2" using ($1*$2):($1*$2*$3*$4):6 with lines title "uncoalesced",\
	"perf.log.3" using ($1*$2):($1*$2*$3*$4):6 with lines title "coalesced"
