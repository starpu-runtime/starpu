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
#include <common/utils.h>
#include <datawizard/datawizard.h>

void starpu_data_set_reduction_methods(starpu_data_handle handle,
					void (*redux_func)(void *, void *),
					void (*init_func)(void *))
{
	_starpu_spin_lock(&handle->header_lock);

	unsigned child;
	for (child = 0; child < handle->nchildren; child++)
	{
		/* make sure that the flags are applied to the children as well */
		struct starpu_data_state_t *child_handle = &handle->children[child];
		if (child_handle->nchildren > 0)
			starpu_data_set_reduction_methods(child_handle, redux_func, init_func);
	}

	handle->redux_func = redux_func;
	handle->init_func = init_func;

	_starpu_spin_unlock(&handle->header_lock);
}


