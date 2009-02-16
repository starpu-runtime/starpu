#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "bench-bandwith-strided.eps"
set title "CUDA Bandwith"
set logscale x
set xlabel "Size (Bytes/4)"
set ylabel "Bandwith (MB/s)"

# plot ".results/htod-pin.data" with linespoint	title "Host to Device (pinned)" ,\
#      ".results/htod-pin.32.data"  with linespoint   title "stride 32" ,\
#      ".results/htod-pin.128.data"  with linespoint   title "stride 128" ,\
#      ".results/htod-pin.512.data"  with linespoint   title "stride 512" ,\
#      ".results/htod-pin.1024.data"  with linespoint   title "stride 1024" ,\
#      ".results/htod-pin.2048.data"  with linespoint   title "stride 2048" ,\
#      ".results/htod-pin.4096.data"  with linespoint   title "stride 4096" ,\
#      ".results/htod-pin.8192.data"  with linespoint   title "stride 8192" 
# 


plot ".results/htod-pin.data" with linespoint	title "Host to Device (pinned)" ,\
     ".results/htod-pin.2.data"  with linespoint   title "stride 2" ,\
     ".results/htod-pin.4.data"  with linespoint   title "stride 4",\
     ".results/htod-pin.8.data"  with linespoint   title "stride 8" ,\
     ".results/htod-pin.16.data"  with linespoint   title "stride 16" ,\
     ".results/htod-pin.32.data"  with linespoint   title "stride 32" ,\
     ".results/htod-pin.64.data"  with linespoint   title "stride 64" ,\
     ".results/htod-pin.128.data"  with linespoint   title "stride 128" ,\
     ".results/htod-pin.256.data"  with linespoint   title "stride 256" ,\
     ".results/htod-pin.512.data"  with linespoint   title "stride 512" ,\
     ".results/htod-pin.1024.data"  with linespoint   title "stride 1024" ,\
     ".results/htod-pin.2048.data"  with linespoint   title "stride 2048" ,\
     ".results/htod-pin.4096.data"  with linespoint   title "stride 4096" ,\
     ".results/htod-pin.8192.data"  with linespoint   title "stride 8192" 


