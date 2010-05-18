/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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
#include <common/config.h>
#include <datawizard/coherency.h>

void _starpu_benchmark_ping_pong(starpu_data_handle handle,
			unsigned node0, unsigned node1, unsigned niter)
{
	/* We assume that no one is using that handle !! */
	unsigned iter;
	for (iter = 0; iter < niter; iter++)
	{
		int ret;
		ret = _starpu_fetch_data_on_node(handle, node0, STARPU_RW, 0);
		STARPU_ASSERT(!ret);
		ret = _starpu_fetch_data_on_node(handle, node1, STARPU_RW, 0);
		STARPU_ASSERT(!ret);
	}
}
