/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2013  Université de Bordeaux 1
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

//! [To be included. You should update doxygen if you see this text.]
static struct starpu_codelet cl11 =
{
	.cpu_funcs = {chol_cpu_codelet_update_u11, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u11, NULL},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1, NULL},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &chol_model_11
};
//! [To be included. You should update doxygen if you see this text.]
