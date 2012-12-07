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
set title "Taux du debit per core normalise"
set grid y                                                                
set grid x                                                                
set xrange [20:86]
set yrange [0.6:1.5]

#set logscale x                                                           
set xtics ("20/76" 20,"30/66" 30,"40/56" 40, "50/46" 50, "60/36" 60, "70/26" 70, "80/16" 80, "86/10" 86)
set key invert box right
#set size 0.1

set xlabel "Nombre de cpus dans le premier contexte / Nombre de cpus dans le deuxieme contexte"
set ylabel "Efficacite per core"     

                                         
plot "res_isole" using 1:5 title 'Gflop rate per core' with lines lt rgb "blue" lw 2
                                                                        
EOF