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

#include <core/dependencies/htable.h>
#include <string.h>

void *_starpu_htbl_search_tag(starpu_htbl_node_t *htbl, starpu_tag_t tag)
{
	unsigned currentbit;
	starpu_htbl_node_t *current_htbl = htbl;

	/* 000000000001111 with STARPU_HTBL_NODE_SIZE 1's */
	starpu_tag_t mask = (1<<STARPU_HTBL_NODE_SIZE)-1;

	for(currentbit = 0; currentbit < TAG_SIZE; currentbit+=STARPU_HTBL_NODE_SIZE)
	{
	
	//	printf("search : current bit = %d \n", currentbit);
		if (STARPU_UNLIKELY(current_htbl == NULL))
			return NULL;

		/* 0000000000001111 
		 *     | currentbit
		 * 0000111100000000 = offloaded_mask
		 *         |last_currentbit
		 * */

		unsigned last_currentbit = 
			TAG_SIZE - (currentbit + STARPU_HTBL_NODE_SIZE);
		starpu_tag_t offloaded_mask = mask << last_currentbit;
		unsigned current_index = 
			(tag & (offloaded_mask)) >> (last_currentbit);

		current_htbl = current_htbl->children[current_index];
	}

	return current_htbl;
}

/*
 * returns the previous value of the tag, or NULL else
 */

void *_starpu_htbl_insert_tag(starpu_htbl_node_t **htbl, starpu_tag_t tag, void *entry)
{

	unsigned currentbit;
	starpu_htbl_node_t **current_htbl_ptr = htbl;
	starpu_htbl_node_t *previous_htbl_ptr = NULL;

	/* 000000000001111 with STARPU_HTBL_NODE_SIZE 1's */
	starpu_tag_t mask = (1<<STARPU_HTBL_NODE_SIZE)-1;

	for(currentbit = 0; currentbit < TAG_SIZE; currentbit+=STARPU_HTBL_NODE_SIZE)
	{
		if (*current_htbl_ptr == NULL) {
			/* TODO pad to change that 1 into 16 ? */
			*current_htbl_ptr = calloc(1, sizeof(starpu_htbl_node_t));
			assert(*current_htbl_ptr);

			if (previous_htbl_ptr)
				previous_htbl_ptr->nentries++;
		}

		/* 0000000000001111 
		 *     | currentbit
		 * 0000111100000000 = offloaded_mask
		 *         |last_currentbit
		 * */

		unsigned last_currentbit = 
			TAG_SIZE - (currentbit + STARPU_HTBL_NODE_SIZE);
		starpu_tag_t offloaded_mask = mask << last_currentbit;
		unsigned current_index = 
			(tag & (offloaded_mask)) >> (last_currentbit);

		previous_htbl_ptr = *current_htbl_ptr;
		current_htbl_ptr = 
			&((*current_htbl_ptr)->children[current_index]);

	}

	/* current_htbl either contains NULL or a previous entry 
	 * we overwrite it anyway */
	void *old_entry = *current_htbl_ptr;
	*current_htbl_ptr = entry;

	if (!old_entry)
		previous_htbl_ptr->nentries++;

	return old_entry;
}

/* returns the entry corresponding to the tag and remove it from the htbl */
void *_starpu_htbl_remove_tag(starpu_htbl_node_t *htbl, starpu_tag_t tag)
{
	/* NB : if the entry is "NULL", we assume this means it is not present XXX */
	unsigned currentbit;
	starpu_htbl_node_t *current_htbl_ptr = htbl;

	/* remember the path to the tag */
	starpu_htbl_node_t *path[(TAG_SIZE + STARPU_HTBL_NODE_SIZE - 1)/(STARPU_HTBL_NODE_SIZE)];

	/* 000000000001111 with STARPU_HTBL_NODE_SIZE 1's */
	starpu_tag_t mask = (1<<STARPU_HTBL_NODE_SIZE)-1;
	int level, maxlevel;
	unsigned tag_is_present = 1;

	for(currentbit = 0, level = 0; currentbit < TAG_SIZE; currentbit+=STARPU_HTBL_NODE_SIZE, level++)
	{
		path[level] = current_htbl_ptr;

		if (STARPU_UNLIKELY(!current_htbl_ptr)) {
			tag_is_present = 0;
			break;
		}

		/* 0000000000001111 
		 *     | currentbit
		 * 0000111100000000 = offloaded_mask
		 *         |last_currentbit
		 * */

		unsigned last_currentbit = 
			TAG_SIZE - (currentbit + STARPU_HTBL_NODE_SIZE);
		starpu_tag_t offloaded_mask = mask << last_currentbit;
		unsigned current_index = 
			(tag & (offloaded_mask)) >> (last_currentbit);
		
		current_htbl_ptr = 
			current_htbl_ptr->children[current_index];
	}

	maxlevel = level;
	if (STARPU_UNLIKELY(!current_htbl_ptr))
		tag_is_present = 0;

	void *old_entry = current_htbl_ptr;

	if (tag_is_present) {
		/* the tag was in the htbl, so we have to unroll the search 
 		 * to remove possibly useless htbl (internal) nodes */
		for (level = maxlevel - 1; level >= 0; level--)
		{
			path[level]->nentries--;

			/* TODO use likely statements ... */

			/* in case we do not remove that node, we do decrease its parents
 			 * number of entries */
			if (path[level]->nentries > 0)
				break;

			/* we remove this node */
			free(path[level]);
		}
	}

	/* we return the entry if there was one */
	return old_entry;
}
