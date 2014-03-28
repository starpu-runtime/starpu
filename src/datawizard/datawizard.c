/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2010, 2012-2014  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2013  Centre National de la Recherche Scientifique
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
#include <common/config.h>
#include <datawizard/datawizard.h>
#include <core/workers.h>
#include <core/progress_hook.h>
#ifdef STARPU_SIMGRID
#include <msg/msg.h>
#endif

int __starpu_datawizard_progress(unsigned memory_node, unsigned may_alloc, unsigned push_requests)
{
	int ret = 0;

#if STARPU_DEVEL
#warning FIXME
#endif
#ifdef STARPU_SIMGRID
	MSG_process_sleep(0.000010);
#endif
	STARPU_UYIELD();

	/* in case some other driver requested data */
	if (_starpu_handle_pending_node_data_requests(memory_node))
		ret = 1;
	if (push_requests)
	{
		unsigned pushed;
		if (_starpu_handle_node_data_requests(memory_node, may_alloc, &pushed) == 0)
		{
			if (pushed)
				ret = 1;
			/* We pushed all pending requests, we can afford pushing
			 * prefetch requests */
			_starpu_handle_node_prefetch_requests(memory_node, may_alloc, &pushed);
		}
		if (pushed)
			ret = 1;
	}
	_starpu_execute_registered_progression_hooks();

	return ret;
}

int _starpu_datawizard_progress(unsigned memory_node, unsigned may_alloc)
{
	__starpu_datawizard_progress(memory_node, may_alloc, 1);
}
