/* this file is intended to be used by both gcc and
 *  spu-gcc so that we don't copy code twice ... */

extern void driver_copy_data(data_state *state, uint32_t src_node_mask, uint32_t dst_node, unsigned donotread);
extern void driver_copy_data_1_to_1(data_state *state, uint32_t node, uint32_t requesting_node, unsigned donotread);
extern unsigned get_local_memory_node(void);

void display_state(data_state *state)
{
	uint32_t node;

	printf("******************************************\n");
	for (node = 0; node < MAXNODES; node++)
	{
		switch(state->per_node[node].state) {
			case INVALID:
				printf("\t%d\tINVALID\n", node);
				break;
			case OWNER:
				printf("\t%d\tOWNER\n", node);
				break;
			case SHARED:
				printf("\t%d\tSHARED\n", node);
				break;
		}
	}

	printf("******************************************\n");

}

/* this function will actually copy a valid data into the requesting node */
static void copy_data_to_node(data_state *state, uint32_t requesting_node, unsigned donotread)
{
	/* first find a valid copy, either a OWNER or a SHARED */
	uint32_t node;
	uint32_t src_node_mask = 0;
	for (node = 0; node < MAXNODES; node++)
	{
		if (state->per_node[node].state != INVALID) {
			/* we found a copy ! */
			src_node_mask |= (1<<node);
		}
	}

	/* we should have found at least one copy ! */
	ASSERT(src_node_mask != 0);

	driver_copy_data(state, src_node_mask, requesting_node, donotread);
	return;
}

/*
 * This function is called when the data is needed on the local node, this
 * returns a pointer to the local copy 
 *
 *			R 	W 	RW
 *	Owner		OK	OK	OK
 *	Shared		OK	1	1
 *	Invalid		2	3	4
 *
 * case 1 : shared + (read)write : 
 * 	no data copy but shared->Invalid/Owner
 * case 2 : invalid + read : 
 * 	data copy + invalid->shared + owner->shared (ASSERT(there is a valid))
 * case 3 : invalid + write : 
 * 	no data copy + invalid->owner + (owner,shared)->invalid
 * case 4 : invalid + R/W : 
 * 	data copy + if (W) (invalid->owner + owner->invalid) else (invalid,owner->shared)
 */
/* NB : for SPU this is a pointer to the local copy which is not entirely fetched at first ! */

uintptr_t _fetch_data(data_state *state, uint32_t requesting_node,
			uint8_t read, uint8_t write)
{
	take_mutex(&state->header_lock);

#ifdef SPU_CODE
	/* we may now fetch the entire state structure 
 	 * from main memory since it is locked */
	fetch_dynamic_data_state(state);	
#endif

//	printf("FETCH from %d R,W = %d,%d\n", requesting_node, read, write);

	cache_state local_state;
	local_state = state->per_node[requesting_node].state;


	/* we handle that case first to optimize the OWNER path */
	if ((local_state == OWNER) || (local_state == SHARED && !write))
	{
		/* the local node already got its data */
		release_mutex(&state->header_lock);

		return state->per_node[requesting_node].ptr;
	}

	if ((local_state == SHARED) && write) {
		/* local node already has the data but it must invalidate other copies */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			if (state->per_node[node].state == SHARED) 
			{
				state->per_node[node].state =
					(node == requesting_node ? OWNER:INVALID);
			}

		}
		
#ifdef SPU_CODE
		/* some states were modified */
		push_dynamic_data_state(state);
#endif

		release_mutex(&state->header_lock);

		return state->per_node[requesting_node].ptr;
	}

	/* the only remaining situation is that the local copy was invalid */
	ASSERT(state->per_node[requesting_node].state == INVALID);

	/* we first need to copy the data from either the owner or one of the sharer */
	copy_data_to_node(state, requesting_node, !read);

	if (write) {
		/* the requesting node now has the only valid copy */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			state->per_node[node].state = INVALID;
		}
		state->per_node[requesting_node].state = OWNER;
	}
	else { /* read only */
		/* there was at least another copy of the data */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			if (state->per_node[node].state != INVALID)
				state->per_node[node].state = SHARED;
		}
		state->per_node[requesting_node].state = SHARED;
	}

#ifdef SPU_CODE
		/* some states were modified */
		push_dynamic_data_state(state);
#endif

	release_mutex(&state->header_lock);

	return state->per_node[requesting_node].ptr;
}

uintptr_t fetch_data(data_state *state, access_mode mode)
{
	uint32_t requesting_node = get_local_memory_node(); 

	uint8_t read, write;
	read = (mode != W); /* then R or RW */
	write = (mode != R); /* then W or RW */

	if (write) {
		take_rw_lock_write(&state->data_lock);
	} else {
		take_rw_lock_read(&state->data_lock);
	}

	take_mutex(&state->header_lock);
	state->per_node[requesting_node].refcnt++;
	release_mutex(&state->header_lock);

	return _fetch_data(state, requesting_node, read, write);
}

uint32_t get_data_refcnt(data_state *state, uint32_t node)
{
	return state->per_node[node].refcnt;
}

void write_through_data(data_state *state, uint32_t requesting_node, uint32_t write_through_mask)
{
	if ((write_through_mask & ~(1<<requesting_node)) == 0) {
		/* nothing will be done ... */
		return;
	}

	take_mutex(&state->header_lock);

	/* first commit all changes onto the nodes specified by the mask */
	uint32_t node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (write_through_mask & (1<<node)) {
			/* we need to commit the buffer on that node */
			if (node != requesting_node) 
			{
//				printf("write_through_data %d -> %d \n", requesting_node, node);
				/* the requesting node already has the data by definition */
				driver_copy_data_1_to_1(state, requesting_node, node, 0);
			}
				
			/* now the data is shared among the nodes on the write_through_mask */
			state->per_node[node].state = SHARED;
		}
	}

	/* the requesting node is now one sharer */
	if (write_through_mask & ~(1<<requesting_node))
	{
		state->per_node[requesting_node].state = SHARED;
	}

	release_mutex(&state->header_lock);
}

/* in case the data was accessed on a write mode, do not forget to 
 * make it accessible again once it is possible ! */
void release_data(data_state *state, uint32_t write_through_mask)
{
	/* normally, the requesting node should have the data in an exclusive manner */
	uint32_t requesting_node = get_local_memory_node();
	ASSERT(state->per_node[requesting_node].state != INVALID);
	
	/* are we doing write-through or just some normal write-back ? */
	if (write_through_mask & ~(1<<requesting_node)) {
		write_through_data(state, requesting_node, write_through_mask);
#ifdef SPU_CODE
		/* some states were (possibly) modified */
		push_dynamic_data_state(state);
#endif
	}

	take_mutex(&state->header_lock);
	state->per_node[requesting_node].refcnt--;
	release_mutex(&state->header_lock);

	/* this is intended to make data accessible again */
	release_rw_lock(&state->data_lock);
}

