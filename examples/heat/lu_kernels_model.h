/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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

#ifndef __LU_KERNELS_MODEL_H__
#define __LU_KERNELS_MODEL_H__

#include <starpu.h>

double task_getrf_cost(struct starpu_task *task, unsigned nimpl);
double task_trsm_ll_cost(struct starpu_task *task, unsigned nimpl);
double task_trsm_ru_cost(struct starpu_task *task, unsigned nimpl);
double task_gemm_cost(struct starpu_task *task, unsigned nimpl);

double task_getrf_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double task_trsm_ll_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double task_trsm_ru_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double task_gemm_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);

double task_getrf_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double task_trsm_ll_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double task_trsm_ru_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double task_gemm_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);

void initialize_lu_kernels_model(struct starpu_perfmodel* model, char * symbol,
		double (*cost_function)(struct starpu_task *, unsigned),
		double (*cpu_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned),
		double (*cuda_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned));

#endif /* __LU_KERNELS_MODEL_H__ */
