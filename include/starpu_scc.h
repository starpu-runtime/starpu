/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015,2017,2019                           CNRS
 * Copyright (C) 2013                                     Université de Bordeaux
 * Copyright (C) 2012                                     Inria
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

#ifndef __STARPU_SCC_H__
#define __STARPU_SCC_H__

#include <starpu_config.h>

/**
   @defgroup API_SCC_Extensions SCC Extensions
   @{
*/

#ifdef STARPU_USE_SCC

/**
   Type for SCC function symbols
*/
typedef void *starpu_scc_func_symbol_t;

/**
   Initiate a lookup on each SCC device to find the adress of the
   function named \p func_name, store them in the global array kernels
   and return the index in the array through \p symbol.
*/
int starpu_scc_register_kernel(starpu_scc_func_symbol_t *symbol, const char *func_name);

/**
   If success, return the pointer to the function defined by \p symbol on
   the device linked to the called device. This can for instance be used
   in a starpu_scc_func_symbol_t implementation.
*/
starpu_scc_kernel_t starpu_scc_get_kernel(starpu_scc_func_symbol_t symbol);

/**
   Assign the offset to \p offset between \p ptr and the start of the
   shared memory.
   Assign \p dev_handle with the start of the shared memory is useful
   for data partionning.
 */
void starpu_scc_get_offset_in_shared_memory(void *ptr, void **dev_handle, size_t *offset);

#endif /* STARPU_USE_SCC */

/** @} */

#endif /* __STARPU_SCC_H__ */
