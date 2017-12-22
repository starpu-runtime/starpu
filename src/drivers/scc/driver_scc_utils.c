/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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


#include <starpu.h>
#include <starpu_scc.h>
#include <common/config.h>

#include <drivers/mp_common/source_common.h>
#include <drivers/scc/driver_scc_source.h>


/* Initiate a lookup on each SCC device to find the adress of the function
 * named FUNC_NAME, store them in the global array kernels and return
 * the index in the array through SYMBOL.
 * If success, returns 0. If the user has registered too many kernels (more
 * than STARPU_MAXSCCDEVS) returns -ENOMEM
 */
int starpu_scc_register_kernel(starpu_scc_func_symbol_t *symbol,
			       const char *func_name)
{
	return _starpu_scc_src_register_kernel(symbol, func_name);
}

/* If success, return the pointer to the function defined by SYMBOL on the
 * device linked to the called 
 * device.
 */
starpu_scc_kernel_t starpu_scc_get_kernel(starpu_scc_func_symbol_t symbol)
{
	return _starpu_scc_src_get_kernel(symbol);
}
