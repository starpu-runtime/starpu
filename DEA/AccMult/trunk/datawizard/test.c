#include "coherency.h"
#include "copy-driver.h"

int main(int argc, char **argv)
{
	init_drivers();
	register_memory_node(RAM);
	register_memory_node(RAM);
	register_memory_node(RAM);
	register_memory_node(GPU_RAM);

	int my_lovely_integer = 0;

	int *val0;//, *val1, *val2;

	data_state my_int_state;
	
	monitor_new_data(&my_int_state, 0 /* home node */,
	     (uintptr_t)&my_lovely_integer, sizeof(my_lovely_integer));

//	/* node 0 reads its own data .. */
//	val0 = (int *)fetch_data(&my_int_state, 0, 0, 1);
//	*val0 = 42;
//	release_data(&my_int_state);
//
////	display_state(&my_int_state);
//
//	/* node 1 reads the data */
//	val1 = (int *)fetch_data(&my_int_state, 1, 1, 0);
//
////	display_state(&my_int_state);
//	/* node 2 reads the data */
//	val2 = (int *)fetch_data(&my_int_state, 2, 1, 0);
//
//	printf("from 2 => %d \n", *val2);
//
//	/* node 2 modifies data .. */
//	val2 = (int *)fetch_data(&my_int_state, 2, 1, 1);
//	*val2 = 1664;
//	release_data(&my_int_state);
//
//	/* node 0 reads data .. */
//	val0 = (int *)fetch_data(&my_int_state, 0, 1, 0);
//	//display_state(&my_int_state);

	uint32_t node;
	for (node = 0; node < MAXNODES; node++)
	{
		int *val;
		val = (int *)fetch_data(&my_int_state, node, 1, 1);
		*val = *val + 1;
		release_data(&my_int_state);
	}

	val0 = (int *)fetch_data(&my_int_state, 0, 1, 0);

	printf("from 0 => %d \n", *val0);
	return 0;
}
