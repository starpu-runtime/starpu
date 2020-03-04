/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __MULTIPLE_REGRESSION_H__
#define __MULTIPLE_REGRESSION_H__

/** @file */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <core/perfmodel/perfmodel.h>
#include <starpu.h>

int _starpu_multiple_regression(struct starpu_perfmodel_history_list *ptr, double *coeff, unsigned ncoeff, unsigned nparameters, const char **parameters_names, unsigned **combinations, const char *codelet_name);

#endif // __MULTIPLE_REGRESSION_H__
