#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "bench-bandwith.eps"
set title "CUDA Bandwith"
set logscale x
set xlabel "Size (Bytes)"
set ylabel "Bandwith (MB/s)"

plot ".results/htod-pin.data" with linespoint	title "Host to Device (pinned)",\
     ".results/dtoh-pin.data" with linespoint   title "Device to Host (pinned)"

