/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>

#ifdef STARPU_USE_GORDON
static inline unsigned load_gordon_null_kernel(void)
{
	unsigned elf_id =
		gordon_register_elf_plugin("./gordon/null_kernel_gordon.spuelf");
	gordon_load_plugin_on_all_spu(elf_id);

	unsigned gordon_null_kernel =
		gordon_register_kernel(elf_id, "empty_kernel");
	gordon_load_kernel_on_all_spu(gordon_null_kernel);

	return gordon_null_kernel;
}
#endif
