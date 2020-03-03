/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_RAND_H__
#define __STARPU_RAND_H__

#include <stdlib.h>
#include <starpu_config.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Random_Functions Random Functions
   @{
 */

#ifdef STARPU_SIMGRID
/* In simgrid mode, force using seed 0 by default to get reproducible behavior by default */
#define starpu_seed(seed)				starpu_get_env_number_default("STARPU_RAND_SEED", 0)
#else
#define starpu_seed(seed)				starpu_get_env_number_default("STARPU_RAND_SEED", (seed))
#endif

#ifdef STARPU_USE_DRAND48
#  define starpu_srand48(seed)				srand48(starpu_seed(seed))
#  define starpu_drand48()				drand48()
#  define starpu_lrand48()				lrand48()
#  define starpu_erand48(xsubi)				erand48(xsubi)
#  ifdef STARPU_USE_ERAND48_R
typedef struct drand48_data starpu_drand48_data;
#    define starpu_srand48_r(seed, buffer)		srand48_r(starpu_seed(seed), buffer)
#    define starpu_drand48_r(buffer, result)		drand48_r(buffer, result)
#    define starpu_lrand48_r(buffer, result)		lrand48_r(buffer, result)
#    define starpu_erand48_r(xsubi, buffer, result)	erand48_r(xsubi, buffer, result)
#else
typedef int starpu_drand48_data;
#    define starpu_srand48_r(seed, buffer)		srand48(starpu_seed(seed))
#    define starpu_drand48_r(buffer, result)		do {*(result) = drand48(); } while (0)
#    define starpu_lrand48_r(buffer, result)		do {*(result) = lrand48(); } while (0)
#    define starpu_erand48_r(xsubi, buffer, result)	do {(void) buffer; *(result) = erand48(xsubi); } while (0)
#  endif
#else
typedef int starpu_drand48_data;
#  define starpu_srand48(seed)				srand(starpu_seed(seed))
#  define starpu_drand48() 				(double)(rand()) / RAND_MAX
#  define starpu_lrand48() 				rand()
#  define starpu_erand48(xsubi)				starpu_drand48()
#  define starpu_srand48_r(seed, buffer) 		srand(starpu_seed(seed))
#  define starpu_erand48_r(xsubi, buffer, result)	do {(void) xsubi; (void) buffer; *(result) = ((double)(rand()) / RAND_MAX);} while (0)
#endif

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_RAND_H__ */
