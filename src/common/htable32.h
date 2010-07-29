/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

#ifndef __GENERIC_HTABLE_H__
#define __GENERIC_HTABLE_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define STARPU_HTBL32_NODE_SIZE	16

typedef struct starpu_htbl32_node_s {
	unsigned nentries;
	struct starpu_htbl32_node_s *children[1<<STARPU_HTBL32_NODE_SIZE];
} starpu_htbl32_node_t;

void *_starpu_htbl_search_32(struct starpu_htbl32_node_s *htbl, uint32_t key);
void *_starpu_htbl_insert_32(struct starpu_htbl32_node_s **htbl, uint32_t key, void *entry);

#endif // __GENERIC_HTABLE_H__
