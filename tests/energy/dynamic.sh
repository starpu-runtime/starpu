#!/bin/sh

# To have 24 cores
export STARPU_HOSTNAME=sirocco

# To avoid slowing down simulation
export MALLOC_PERTURB_=0

# You can play with these
export N=40
export NITER=30

GAMMAS="1000000 100000 76000 10000 0"

for gamma in $GAMMAS; do
	(for freq_slow in $(seq 1200 200 3500) ; do 
		STARPU_SCHED_GAMMA=$gamma STARPU_FREQ_SLOW=$freq_slow \
			./energy_efficiency $N $NITER | grep "^$(($N * 512))	" &
	done) | sort -n -k 2 > dynamic.$gamma.dat
done

cat > dynamic.gp << EOF
set output "dynamic.eps"
set term postscript eps enhanced color font ",20"
set key bottom right
set xlabel "performance (GFlop/s)"
set ylabel "energy (J)"

plot \\
EOF
for gamma in $GAMMAS; do
	cat >> dynamic.gp << EOF
	"dynamic.$gamma.dat" using 5:7:6:8 with xyerrorlines lw 2 title "$gamma", \\
EOF
done

cat >> dynamic.gp << EOF

set output "dynamic-time.eps"
set xlabel "time (ms)"
set ylabel "energy (J)"

plot \\
EOF
for gamma in $GAMMAS; do
	cat >> dynamic.gp << EOF
	"dynamic.$gamma.dat" using 3:7:4:8 with xyerrorlines lw 2 title "$gamma", \\
EOF
done


gnuplot dynamic.gp
gv dynamic.eps &
gv dynamic-time.eps &
