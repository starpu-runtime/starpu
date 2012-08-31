/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2010, 2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include <common/htable64.h>
#include <stdint.h>
#include <string.h>

void *_starpu_htbl_search_64(struct starpu_htbl64_node *htbl, uint64_t key)
{
	unsigned currentbit;
	unsigned keysize = sizeof(uint64_t)*8;

	struct starpu_htbl64_node *current_htbl = htbl;
	uint64_t mask = (1ULL<<_STARPU_HTBL64_NODE_SIZE)-1;

	for(currentbit = 0; currentbit < keysize; currentbit+=_STARPU_HTBL64_NODE_SIZE)
	{
		if (STARPU_UNLIKELY(current_htbl == NULL))
			return NULL;

		unsigned last_currentbit =
			keysize - (currentbit + _STARPU_HTBL64_NODE_SIZE);
		uint64_t offloaded_mask = mask << last_currentbit;
		unsigned current_index =
			(key & (offloaded_mask)) >> (last_currentbit);

		current_htbl = current_htbl->children[current_index];
	}
	return current_htbl;
}

/*
 * returns the previous value of the tag, or NULL else
 */

void *_starpu_htbl_insert_64(struct starpu_htbl64_node **htbl, uint64_t key, void *entry)
{
	unsigned currentbit;
	unsigned keysize = sizeof(uint64_t)*8;
	struct starpu_htbl64_node **current_htbl_ptr = htbl;

	uint64_t mask = (1ULL<<_STARPU_HTBL64_NODE_SIZE)-1;
	for(currentbit = 0; currentbit < keysize; currentbit+=_STARPU_HTBL64_NODE_SIZE)
	{
		if (*current_htbl_ptr == NULL)
		{
			*current_htbl_ptr = (struct starpu_htbl64_node*)calloc(sizeof(struct starpu_htbl64_node), 1);
			STARPU_ASSERT(*current_htbl_ptr);
		}

		unsigned last_currentbit =
			keysize - (currentbit + _STARPU_HTBL64_NODE_SIZE);
		uint64_t offloaded_mask = mask << last_currentbit;
		unsigned current_index =
			(key & (offloaded_mask)) >> (last_currentbit);

		current_htbl_ptr =
			&((*current_htbl_ptr)->children[current_index]);
	}
	void *old_entry = *current_htbl_ptr;
	*current_htbl_ptr = (struct starpu_htbl64_node *) entry;

	return old_entry;
}

static void _starpu_htbl_destroy_64_bit(struct starpu_htbl64_node *htbl, unsigned bit, void (*remove)(void*))
{
	unsigned keysize = sizeof(uint64_t)*8;
	unsigned i;

	if (!htbl)
		return;

	if (bit >= keysize) {
		/* entry, delete it */
		if (remove)
			remove(htbl);
		return;
	}

	for (i = 0; i < 1ULL<<_STARPU_HTBL64_NODE_SIZE; i++) {
		_starpu_htbl_destroy_64_bit(htbl->children[i], bit+_STARPU_HTBL64_NODE_SIZE, remove);
	}

	free(htbl);
}
void _starpu_htbl_destroy_64(struct starpu_htbl64_node *htbl, void (*remove)(void*))
{
	_starpu_htbl_destroy_64_bit(htbl, 0, remove);
}
