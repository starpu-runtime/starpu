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
set term postscript eps enhanced color font ",20"
set key top left
set xlabel "frequency (MHz)"

freq_min = 1200
freq_fast = 3500
power_min = 2
power_fast = 8.2
TRSM_DECAY = 0.5
POTRF_DECAY = 0.5


# Plot the power according to frequency (cubic curve)

freq_min3 = freq_min * freq_min * freq_min
freq_fast3 = freq_fast * freq_fast * freq_fast
alpha = (power_fast - power_min) / (freq_fast3 - freq_min3)
power(frequency) = power_min + alpha * (frequency*frequency*frequency - freq_min3)
 
set output "power.eps"
set ylabel "power (W)"

plot [frequency=freq_min:freq_fast] [y=0:] power(frequency) lw 2 notitle


# Plot the kernel performance according to frequency

set output "perfs.eps"
set ylabel "performance (GFlop/s)"

gemm_max_perf = 50
trsm_max_perf = 35.784040
potrf_max_perf = 6.964803

gemm_factor(frequency) = frequency / freq_fast
trsm_factor(frequency) = (frequency - freq_min/2) ** TRSM_DECAY / (freq_fast - freq_min/2) ** TRSM_DECAY
potrf_factor(frequency) = 1 - POTRF_DECAY * ((freq_min/(frequency-freq_min/2)) - (freq_min/(freq_fast-freq_min/2)))

plot [frequency=freq_min:freq_fast] \
     gemm_max_perf * gemm_factor(frequency) lw 2 title "gemm", \
     trsm_max_perf * trsm_factor(frequency) lw 2 title "trsm", \
     potrf_max_perf * potrf_factor(frequency) lw 2 title "potrf"


# Plot the kernel efficiency according to frequency

set output "efficiency.eps"
set key top right
set ylabel "efficiency (GFlop/W)"

gemm_max_efficiency = 6.097561
trsm_max_efficiency = 4.363907
potrf_max_efficiency = 0.849366

power_factor(frequency) = power(frequency) / power(freq_fast)

plot [frequency=freq_min:freq_fast] \
     gemm_max_efficiency * gemm_factor(frequency) / power_factor(frequency) lw 2 title "gemm", \
     trsm_max_efficiency * trsm_factor(frequency) / power_factor(frequency)  lw 2 title "trsm", \
     potrf_max_efficiency * potrf_factor(frequency) / power_factor(frequency)  lw 2 title "potrf"

