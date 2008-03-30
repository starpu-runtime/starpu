#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>

#include <datawizard/coherency.h>

data_state my_int_state;

void callback_func(void *argcb)
{
	//unsigned node;
	//node = get_local_memory_node();
	//printf("callback core on node %d\n", node);
}

void core_codelet(void *_args)
{
	unsigned node;
	node = get_local_memory_node();
	//printf("codelet core on node %d\n", node);
	
	int *val;
	val = (int *)fetch_data(&my_int_state, node, 1, 1);
	*val = *val +1;

	printf("val = %d\n", *val);
	release_data(&my_int_state, node, 0);
}

void cublas_codelet(void *_args)
{
	unsigned node;
	node = get_local_memory_node();
	//printf("codelet cublas on node %d\n", node);

	int *val;
	val = (int *)fetch_data(&my_int_state, node, 1, 1);
	*val = *val +1;
	release_data(&my_int_state, node, 0);
}

int main(int argc, char **argv)
{
	init_machine();
	init_workers();

	uint64_t my_lovely_integer = 0;

	monitor_new_data(&my_int_state, 0 /* home node */,
		(uintptr_t)&my_lovely_integer, sizeof(my_lovely_integer));

//	init_memory_nodes();

	printf("ok\n");

	codelet cl;

	job_t j;

	unsigned i;
	for (i = 0; i < 1000000; i++)
	{
		j = job_new();
		j->type = CODELET;
		j->where = ANY;
		j->cb = callback_func;
		j->argcb = NULL;
		j->cl = &cl;

		cl.cl_arg = NULL;
		cl.core_func = core_codelet;
		cl.cublas_func = cublas_codelet;

		push_task(j);
	}

	sleep(100);
}
