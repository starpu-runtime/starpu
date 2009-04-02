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

#define HTBL_NODE_SIZE	16

typedef struct _htbl_node_t {
	unsigned nentries;
	struct _htbl_node_t *children[1<<HTBL_NODE_SIZE];
} htbl_node_t;

void *htbl_search_tag(htbl_node_t *htbl, starpu_tag_t tag);
void *htbl_insert_tag(htbl_node_t **htbl, starpu_tag_t tag, void *entry);
void *htbl_remove_tag(htbl_node_t *htbl, starpu_tag_t tag);


#endif
