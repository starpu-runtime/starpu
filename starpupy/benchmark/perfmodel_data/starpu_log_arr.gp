#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "starpu_log_arr.eps"
set title "Model for codelet log-arr on he-XPS-13-9370"
set xlabel "Total data size"
set ylabel "Time (ms)"

set key top left
set logscale x
set logscale y

set xrange [1 < * < 10**5 : 10**6 < * < 10**9]

plot	".//starpu_log_arr_avg.data" using 1:2:3 with errorlines title "Average cpu0-impl0 (Comb0)"
