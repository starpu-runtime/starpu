#include "htable.h"
#include "string.h"

void *htbl_search_tag(htbl_node_t *htbl, tag_t tag)
{
	unsigned currentbit;
	htbl_node_t *current_htbl = htbl;

	/* 000000000001111 with HTBL_NODE_SIZE 1's */
	tag_t mask = (1<<HTBL_NODE_SIZE)-1;

	for(currentbit = 0; currentbit < TAG_SIZE; currentbit+=HTBL_NODE_SIZE)
	{
	
	//	printf("search : current bit = %d \n", currentbit);
		if (current_htbl == NULL)
			return NULL;

		/* 0000000000001111 
		 *     | currentbit
		 * 0000111100000000 = offloaded_mask
		 *         |last_currentbit
		 * */

		unsigned last_currentbit = 
			TAG_SIZE - (currentbit + HTBL_NODE_SIZE);
		tag_t offloaded_mask = mask << last_currentbit;
		unsigned current_index = 
			(tag & (offloaded_mask)) >> (last_currentbit);

		current_htbl = current_htbl->children[current_index];
	}

	return current_htbl;
}

/*
 * returns the previous value of the tag, or NULL else
 */

void *htbl_insert_tag(htbl_node_t **htbl, tag_t tag, void *entry)
{
	unsigned currentbit;
	htbl_node_t **current_htbl_ptr = htbl;

	/* 000000000001111 with HTBL_NODE_SIZE 1's */
	tag_t mask = (1<<HTBL_NODE_SIZE)-1;

	for(currentbit = 0; currentbit < TAG_SIZE; currentbit+=HTBL_NODE_SIZE)
	{
		//printf("insert : current bit = %d \n", currentbit);
		if (*current_htbl_ptr == NULL) {
			/* TODO pad to change that 1 into 16 ? */
			*current_htbl_ptr = calloc(sizeof(htbl_node_t), 1);
			assert(*current_htbl_ptr);
		}

		/* 0000000000001111 
		 *     | currentbit
		 * 0000111100000000 = offloaded_mask
		 *         |last_currentbit
		 * */

		unsigned last_currentbit = 
			TAG_SIZE - (currentbit + HTBL_NODE_SIZE);
		tag_t offloaded_mask = mask << last_currentbit;
		unsigned current_index = 
			(tag & (offloaded_mask)) >> (last_currentbit);

		current_htbl_ptr = 
			&((*current_htbl_ptr)->children[current_index]);
	}

	/* current_htbl either contains NULL or a previous entry 
	 * we overwrite it anyway */
	void *old_entry = *current_htbl_ptr;
	*current_htbl_ptr = entry;

	return old_entry;
}
