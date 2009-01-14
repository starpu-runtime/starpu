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

void *htbl_search_tag(htbl_node_t *htbl, tag_t tag);
void *htbl_insert_tag(htbl_node_t **htbl, tag_t tag, void *entry);
void *htbl_remove_tag(htbl_node_t *htbl, tag_t tag);


#endif
