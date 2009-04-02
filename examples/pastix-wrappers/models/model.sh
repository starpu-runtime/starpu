#!/bin/bash

#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#


rm -f generated_model.h

# contrib compact 
./reg_gemm /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.core.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.core.debug` 2> /dev/null|sed -s s/GEMM/GEMM_CPU/ >  generated_model.h
./reg_gemm /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.cuda.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.cuda.debug` 2> /dev/null|sed -s s/GEMM/GEMM_GPU/ >>  generated_model.h

# strsm

./reg_trsm /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.cuda.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.cuda.debug` 2> /dev/null|sed -s s/TRSM/TRSM_GPU/ >>  generated_model.h
./reg_trsm /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.core.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.core.debug` 2> /dev/null|sed -s s/TRSM/TRSM_CPU/ >>  generated_model.h

cat generated_model.h
