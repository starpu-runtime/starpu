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

#export STARPU_NCUDA=3
#export STARPU_NCPUS=9
#export STARPU_DIR=$HOME/sched_ctx/build

#source sched.sh isole 0 0 0 
#source sched_no_ctxs.sh
source sched_no_ctxs.sh 1stchole -chole1
source sched_no_ctxs.sh 2ndchole -chole2
 
source sched_with_ctxs.sh isole 0 0 3 
source sched_with_ctxs.sh isole 0 1 2
source sched_with_ctxs.sh isole 0 2 1
source sched_with_ctxs.sh isole 0 3 0   

source sched_with_ctxs.sh 1gpu 1 0 2
source sched_with_ctxs.sh 1gpu 1 1 1
source sched_with_ctxs.sh 1gpu 1 2 0

source sched_with_ctxs.sh 2gpu 2 1 0
source sched_with_ctxs.sh 2gpu 2 0 1

source sched_with_ctxs.sh 3gpu 3 0 0
