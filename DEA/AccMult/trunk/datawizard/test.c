#include "coherency.h"
#include "copy-driver.h"

int main(int argc, char **argv)
{
	uint32_t node;
	uint32_t turn;

	init_memory_nodes();
	register_memory_node(RAM);
	register_memory_node(RAM);
	register_memory_node(RAM);
	register_memory_node(RAM);
//	register_memory_node(CUBLAS_RAM);

	uint64_t my_lovely_integer = 0;

	int *val0;//, *val1, *val2;

	data_state my_int_state;
	
	monitor_new_data(&my_int_state, 0 /* home node */,
	     (uintptr_t)&my_lovely_integer, sizeof(my_lovely_integer));

	for (turn = 0; turn < 100000000; turn++)
	{
//		for (node = 0; node < MAXNODES; node++)
//		{
			int *val;
		//	uint32_t mask = 1 | (1<<1) | (1<<2) | (1<<3);
			uint32_t mask = rand() % (1<<4);
			node = rand() % 4;
			val = (int *)fetch_data(&my_int_state, node, 1, 1);
			*val = *val + 1;
//			printf("INC\n");
//			display_state(&my_int_state);
//			printf("RELEASE\n ");
			release_data(&my_int_state, node, mask);
//			display_state(&my_int_state);
//		}
	}

	val0 = (int *)fetch_data(&my_int_state, 0, 1, 0);

	printf("from 0 => %d \n", *val0);
	return 0;
}
