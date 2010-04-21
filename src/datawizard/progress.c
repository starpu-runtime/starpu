/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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
#include <datawizard/datawizard.h>
#include <core/workers.h>

void _starpu_datawizard_progress(uint32_t memory_node, unsigned may_alloc)
{
	/* in case some other driver requested data */
	_starpu_handle_pending_node_data_requests(memory_node);
	_starpu_handle_node_data_requests(memory_node, may_alloc);

	_starpu_execute_registered_progression_hooks();
}
