#!/bin/bash

# StarPU --- Runtime system for heterogeneous multicore architectures.
# 
# Copyright (C) 2011  INRIA
# 
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
# 
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# 
# See the GNU Lesser General Public License in COPYING.LGPL for more details.


filename=$1
gnuplot > /dev/null << EOF                                                
set terminal postscript
set output "| ps2pdf - $filename.pdf"
                                                                   
set datafile missing 'x'                                                  
                                                                          
set pointsize 0.75                                                        
set title "Efficacite de la machine"
set grid y                                                                
set grid x                                                                
set xrange [1:8]
set yrange [0.7:1.7]

#set logscale x                                                           
set xtics ("1/8" 1,"2/7" 2, "3/6" 3,"4/5" 4,"5/4" 5,"6/3" 6,"7/2" 7,"8/1" 8)
set key invert box right
#set size 0.65    

set xlabel "Nombre de cpus dans le premier contexte / Nombre de cpus dans le deuxieme contexte"
set ylabel "Efficacite"        

                                                         
plot "res_isole12" using 4:7 title 'Contextes isoles: 1GPU/2GPU' with lines lt 1 lw 4 lc rgb "blue" ,"res_isole21" using 4:7 title 'Contextes isoles: 2GPU/1GPU' with lines lt 1 lw 4 lc rgb "red" ,"res_isole30" using 4:7 title 'Contextes isoles: 3GPU/0GPU ' with lines lt 1 lw 4 lc rgb "green","res_isole03" using 4:7 title 'Contextes isoles: 0GPU/3GPU' with lines lt 1 lw 4 lc rgb "yellow","res_1gpu02" using 4:7 title 'Partage 1 GPU: 0GPU/2GPU' with lines lt 2 lw 4 lc rgb "pink","res_1gpu11" using 4:7 title 'Partage 1 GPU: 1GPU/1GPU' with lines lt 2 lw 4 lc rgb "aquamarine", "res_1gpu20" using 4:7 title 'Partage 1 GPU: 2GPU/0GPU ' with lines lt 2 lw 4 lc rgb "grey","res_2gpu01" using 4:7 title 'Partage 2 GPU: 0GPU/1 GPU' with lines lt 3 lw 4 lc rgb "brown","res_2gpu10" using 4:7 title 'Partage 2 GPU: 1 GPU/0GPU' with lines lt 3 lw 4 lc  rgb "greenyellow", "res_3gpu00" using 4:7 title 'Partage 3 GPU' with lines lt 4 lw 4 lc  rgb "black"
                                                                        
EOF

gnuplot > /dev/null << EOF                                                
set terminal postscript
set output "| ps2pdf - gflops_$filename.pdf"
                                                                   
set datafile missing 'x'                                                  
                                                                          
set pointsize 0.75                                                        
set title "Efficacite de la machine"
set grid y                                                                
set grid x                                                                
set xrange [1:8]
#set yrange [0.6:2.3]

#set logscale x                                                           
set xtics ("1/8" 1,"2/7" 2, "3/6" 3,"4/5" 4,"5/4" 5,"6/3" 6,"7/2" 7,"8/1" 8)
set key invert box right
#set size 0.65    

set xlabel "Nombre de cpus dans le premier contexte / Nombre de cpus dans le deuxieme contexte"
set ylabel "Gflops/s"        


plot "res_isole12" using 4:6 title 'Contextes isoles et 1GPU/2GPU' with lines lt 1 lw 4 lc rgb "blue" ,"res_isole21" using 4:6 title 'Contextes isoles et 2GPU/1GPU' with lines lt 1 lw 4 lc rgb "red" ,"res_isole30" using 4:6 title 'Contextes isoles et 3GPU/ ' with lines lt 1 lw 4 lc rgb "green","res_isole03" using 4:6 title 'Contextes isoles et /3GPU' with lines lt 1 lw 4 lc rgb "yellow","res_1gpu02" using 4:6 title 'Partage 1 GPU: / 2GPU' with lines lt 2 lw 4 lc rgb "pink","res_1gpu11" using 4:6 title 'Partage 1 GPU: 1GPU/1GPU' with lines lt 2 lw 4 lc rgb "magenta", "res_1gpu20" using 4:6 title 'Partage 1 GPU: 2GPU/ ' with lines lt 2 lw 4 lc rgb "grey","res_2gpu01" using 4:6 title 'Partage 2 GPU: /1 GPU' with lines lt 3 lw 4 lc rgb "orange","res_2gpu10" using 4:6 title 'Partage 2 GPU: 1 GPU/' with lines lt 3 lw 4 lc  rgb "violet", "res_3gpu00" using 4:6 title 'Partage 3 GPU' with lines lt 4 lw 4 lc  rgb "black"
                                                                                                                                 
EOF
