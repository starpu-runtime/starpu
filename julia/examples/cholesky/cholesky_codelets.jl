# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020, 2022       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

chol_model_potrf = starpu_perfmodel(
    perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
    symbol = "chol_model_potrf"
)

chol_model_trsm = starpu_perfmodel(
    perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
    symbol = "chol_model_trsm"
)

chol_model_gemm = starpu_perfmodel(
    perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
    symbol = "chol_model_gemm"
)

cl_potrf = starpu_codelet(
    cpu_func = "potrf",
    cuda_func = "potrf",
    modes = [STARPU_RW],
    color = 0xffff00,
    perfmodel = chol_model_potrf
)
cl_trsm = starpu_codelet(
    cpu_func = "trsm",
    cuda_func = "trsm",
    modes = [STARPU_R, STARPU_RW],
    color = 0x8080ff,
    perfmodel = chol_model_trsm
)
cl_gemm = starpu_codelet(
    cpu_func = "gemm",
    cuda_func = "gemm",
    modes = [STARPU_R, STARPU_R, STARPU_RW],
    color = 0x00ff00,
    perfmodel = chol_model_gemm
)
