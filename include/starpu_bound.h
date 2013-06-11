/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
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

#ifndef __STARPU_BOUND_H__
#define __STARPU_BOUND_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

void starpu_bound_start(int deps, int prio);
void starpu_bound_stop(void);

void starpu_bound_print_dot(FILE *output);

void starpu_bound_compute(double *res, double *integer_res, int integer);

void starpu_bound_print_lp(FILE *output);
void starpu_bound_print_mps(FILE *output);
void starpu_bound_print(FILE *output, int integer);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_BOUND_H__ */
