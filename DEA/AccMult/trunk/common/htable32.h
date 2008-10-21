#ifndef __GENERIC_HTABLE_H__
#define __GENERIC_HTABLE_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define HTBL32_NODE_SIZE	16

typedef struct htbl32_node_s {
	unsigned nentries;
	struct htbl32_node_s *children[1<<HTBL32_NODE_SIZE];
} htbl32_node_t;

void *htbl_search_32(struct htbl32_node_s *htbl, uint32_t key);
void *htbl_insert_32(struct htbl32_node_s **htbl, uint32_t key, void *entry);

#endif // __GENERIC_HTABLE_H__
