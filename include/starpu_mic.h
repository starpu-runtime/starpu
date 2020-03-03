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

#ifndef __STARPU_MIC_H__
#define __STARPU_MIC_H__

#include <starpu_config.h>

#ifdef STARPU_USE_MIC

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_MIC_Extensions MIC Extensions
   @{
*/

/**
   Type for MIC function symbols
*/
typedef void *starpu_mic_func_symbol_t;

/**
   Initiate a lookup on each MIC device to find the address of the
   function named \p func_name, store it in the global array kernels
   and return the index in the array through \p symbol.
*/
int starpu_mic_register_kernel(starpu_mic_func_symbol_t *symbol, const char *func_name);

/**
   If successfull, return the pointer to the function defined by \p symbol on
   the device linked to the called device. This can for instance be used
   in a starpu_mic_func_t implementation.
*/
starpu_mic_kernel_t starpu_mic_get_kernel(starpu_mic_func_symbol_t symbol);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_MIC */

#endif /* __STARPU_MIC_H__ */
