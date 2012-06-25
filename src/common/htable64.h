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

#ifndef __GENERIC_HTABLE_H__
#define __GENERIC_HTABLE_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define _STARPU_HTBL64_NODE_SIZE	8 

/* Hierarchical table: all nodes have a 2^32 arity . */
struct starpu_htbl64_node {
	unsigned nentries;
	struct starpu_htbl64_node *children[1ULL<<_STARPU_HTBL64_NODE_SIZE];
};

/* Look for a 64bit key into the hierchical table. Returns the entry if
 * something is found, NULL otherwise. */
void *_starpu_htbl_search_64(struct starpu_htbl64_node *htbl, uint64_t key);

/* Insert an entry indexed by the 64bit key into the hierarchical table.
 * Returns the entry that was previously associated to that key if any, NULL
 * otherwise. */
void *_starpu_htbl_insert_64(struct starpu_htbl64_node **htbl, uint64_t key, void *entry);

/* Delete the content of the table, `remove' being called on each element */
void _starpu_htbl_destroy_64(struct starpu_htbl64_node *htbl, void (*remove)(void*));

#endif // __GENERIC_HTABLE_H__
