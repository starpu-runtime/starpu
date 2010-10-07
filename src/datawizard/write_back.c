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

#include <datawizard/datawizard.h>

void _starpu_write_through_data(starpu_data_handle handle, uint32_t requesting_node, 
					   uint32_t write_through_mask)
{
	if ((write_through_mask & ~(1<<requesting_node)) == 0) {
		/* nothing will be done ... */
		return;
	}

	while (_starpu_spin_trylock(&handle->header_lock))
		_starpu_datawizard_progress(requesting_node, 1);

	/* first commit all changes onto the nodes specified by the mask */
	uint32_t node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (write_through_mask & (1<<node)) {
			/* we need to commit the buffer on that node */
			if (node != requesting_node) 
			{
				uint32_t handling_node =
					_starpu_select_node_to_handle_request(requesting_node, node);

				starpu_data_request_t r;

				/* check that there is not already a similar
				 * request that we should reuse */
				r = _starpu_search_existing_data_request(&handle->per_node[node], STARPU_R);
				if (!r) {
					/* there was no existing request so we create one now */
					r = _starpu_create_data_request(handle, &handle->per_node[requesting_node],
							&handle->per_node[node], handling_node, STARPU_R, 1);
					_starpu_post_data_request(r, handling_node);
				}
				else {
					/* if there is already a similar request, it is
					 * useless to post another one */
					_starpu_spin_unlock(&r->lock);
				}
			}
		}
	}

	_starpu_spin_unlock(&handle->header_lock);
}

void starpu_data_set_wt_mask(starpu_data_handle handle, uint32_t wt_mask)
{
	handle->wt_mask = wt_mask;

	/* in case the data has some children, set their wt_mask as well */
	if (handle->nchildren > 0) 
	{
		unsigned child;
		for (child = 0; child < handle->nchildren; child++)
			starpu_data_set_wt_mask(&handle->children[child], wt_mask);
	}
}
