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


#ifndef __STARPU_MIC_H__
#define __STARPU_MIC_H__

#include <starpu_config.h>


#ifdef STARPU_USE_MIC

typedef void *starpu_mic_func_symbol_t;

int starpu_mic_register_kernel(starpu_mic_func_symbol_t *symbol, const char *func_name);

starpu_mic_kernel_t starpu_mic_get_kernel(starpu_mic_func_symbol_t symbol);

#endif /* STARPU_USE_MIC */


#endif /* __STARPU_MIC_H__ */
