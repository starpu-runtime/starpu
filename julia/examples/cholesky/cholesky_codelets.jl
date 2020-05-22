# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

chol_model11 = starpu_perfmodel(
    perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
    symbol = "chol_model11"
)

chol_model21 = starpu_perfmodel(
    perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
    symbol = "chol_model21"
)

chol_model22 = starpu_perfmodel(
    perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
    symbol = "chol_model22"
)

cl_11 = starpu_codelet(
    cpu_func = "u11",
    # This kernel cannot be translated to CUDA yet.
    # cuda_func = "u11",
    modes = [STARPU_RW],
    color = 0xffff00,
    perfmodel = chol_model11
)
cl_21 = starpu_codelet(
    cpu_func = "u21",
    cuda_func = "u21",
    modes = [STARPU_R, STARPU_RW],
    color = 0x8080ff,
    perfmodel = chol_model21
)
cl_22 = starpu_codelet(
    cpu_func = "u22",
    cuda_func = "u22",
    modes = [STARPU_R, STARPU_R, STARPU_RW],
    color = 0x00ff00,
    perfmodel = chol_model22
)
