/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __MPI_TESTS_GEMM_HELPER__
#define __MPI_TESTS_GEMM_HELPER__

#include <starpu_config.h>

extern unsigned nslices;
extern unsigned matrix_dim;
extern unsigned check;
extern int comm_thread_cpuid;


void gemm_alloc_data();
int gemm_init_data();
int gemm_submit_tasks();
void gemm_release();
void gemm_add_polling_dependencies();
int gemm_submit_tasks_with_tags(int with_tags);

#endif /* __MPI_TESTS_GEMM_HELPER__ */
