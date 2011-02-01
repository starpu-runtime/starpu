/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
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

#ifndef __HTABLE_H__
#define __HTABLE_H__

/*
 *	Define a hierarchical table to do the tag matching
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <core/dependencies/tags.h>

#define STARPU_HTBL_NODE_SIZE	16

typedef struct starpu_htbl_node_s {
	unsigned nentries;
	struct starpu_htbl_node_s *children[1<<STARPU_HTBL_NODE_SIZE];
} starpu_htbl_node_t;

void *_starpu_htbl_search_tag(starpu_htbl_node_t *htbl, starpu_tag_t tag);
void *_starpu_htbl_insert_tag(starpu_htbl_node_t **htbl, starpu_tag_t tag, void *entry);
void *_starpu_htbl_remove_tag(starpu_htbl_node_t *htbl, starpu_tag_t tag);


#endif
