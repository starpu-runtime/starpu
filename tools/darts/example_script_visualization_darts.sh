# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# Script that launch an experiment with darts and then plot a visualization of the execution.
# Requirements for a visualization:
# 1. Configure with the options --enable-darts-stats --enable-darts-verbose
# 2. Use the following environemnt variable: PRINT_N=$((N)) with N the side of the matrix used in the application
# 3. To launch this script from the starpu/ folder use: 
# bash tools/darts/example_script_visualization_darts.sh $PATH_TO_STARPU_FOLDER N Application NGPU scheduler block_size memory_limitation_of_the_gpus $PATH_TO_PERFMODEL (optional, can also be left empty if the experiment is not done in simulation) hostname (optional if you are not using simulation)
# For instance it can be: bash tools/darts/example_script_visualization_darts.sh /home/name/ 15 Cholesky 1 darts 960 2000 /home/name/starpu/tools/perfmodels/sampling/ attila
# 4. If your targeted application is Cholesky, use -niter 1
# 5. If your targeted application is Gemm, use -iter 1
# The output image will be saved in the starpu/ folder

make -j 6
PATH_STARPU=$1
N=$2
DOSSIER=$3
NGPU=$4
ORDO=$5
block_size=$6
CM=$7
OUTPUT_PATH="/tmp/"
SAVE_DIRECTORY=""
if (( $# > 7 ));
then
    echo "simulation"
    export STARPU_PERF_MODEL_DIR=$8
    HOST=$9
else
    echo "no simulation"
fi
ulimit -S -s 5000000

if [ $DOSSIER = "Matrice_ligne" ]
then
    STARPU_SCHED_OUTPUT=${OUTPUT_PATH} STARPU_SCHED=${ORDO} PRINT_IN_TERMINAL=1 PRINT_N=$((N)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=5 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((block_size*N)) -nblocks $((N)) -iter 1

    python3 ${PATH_STARPU}/starpu/tools/darts/visualization_darts.py ${N} ${ORDO} ${NGPU} ${DOSSIER} 1 ${CM} ${block_size} ${OUTPUT_PATH}
fi

if [ $DOSSIER = "Cholesky" ]
then
    STARPU_SCHED_OUTPUT=${OUTPUT_PATH} STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} PRINT_N=$((N)) DOPT_SELECTION_ORDER=7 STARPU_LIMIT_CUDA_MEM=$((CM)) DEPENDANCES=1 APP=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=${ORDO} STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((block_size*N)) -nblocks $((N)) -niter 1

    python3 ${PATH_STARPU}/starpu/tools/darts/visualization_darts.py ${N} ${ORDO} ${NGPU} ${DOSSIER} 1 ${CM} ${block_size} ${OUTPUT_PATH}
fi
