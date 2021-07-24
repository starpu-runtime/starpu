#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#

# To have 24 cores
export STARPU_HOSTNAME=sirocco

# To avoid slowing down simulation
export MALLOC_PERTURB_=0

# You can play with these
export STARPU_FREQ_SLOW=1200
export STARPU_POWER_SLOW=2
export STARPU_POWER_FAST=8.2
export N=40
export NITER=30

GAMMAS="1000000 100000 10000 0"

for gamma in $GAMMAS; do
	(for ncpu_slow in $(seq 0 24) ; do 
		STARPU_SCHED_GAMMA=$gamma STARPU_NCPU_SLOW=$ncpu_slow \
			./energy_efficiency $N $NITER | grep "^$(($N * 512))	" &
	done) | sort -n -k 2 > static.$gamma.dat
done

cat > static.gp << EOF
set output "static.eps"
set term postscript eps enhanced color font ",20"
set key top center
set xlabel "performance (GFlop/s)"
set ylabel "energy (J)"

plot \\
EOF
for gamma in $GAMMAS; do
	cat >> static.gp << EOF
	"static.$gamma.dat" using 5:7:6:8 with xyerrorlines title "$gamma", \\
EOF
done

cat >> static.gp << EOF

set output "static-time.eps"
set xlabel "time (ms)"
set ylabel "energy (J)"

plot \\
EOF
for gamma in $GAMMAS; do
	cat >> static.gp << EOF
	"static.$gamma.dat" using 3:7:4:8 with xyerrorlines title "$gamma", \\
EOF
done


gnuplot static.gp
gv static.eps &
gv static-time.eps &
