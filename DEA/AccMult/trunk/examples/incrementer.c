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

#define NITER	1000000

data_state my_float_state;
data_state unity_state;

float my_lovely_float[3] = {0.0f, 0.0f, 0.0f};

float unity[3] = {1.0f, 0.0f, 1.0f};
float *dunity;

void callback_func(void *argcb)
{
	//unsigned node;
	//node = get_local_memory_node();
	//printf("callback core on node %d\n", node);
//	if ((int)my_lovely_float[0] % 1000 == 0) {
//	}

	if ((int)my_lovely_float[0] == NITER) 
	{
		printf("-> %f, %f, %f \n", my_lovely_float[0], my_lovely_float[1], my_lovely_float[2]);
		printf("stopping ...\n");
		exit(0);
	}
}

void core_codelet(void *_args)
{
	unsigned node;
	node = get_local_memory_node();
	//printf("codelet core on node %d\n", node);
	

	float *val;
	val = (float *)fetch_data(&my_float_state, node, 1, 1);
	val[0] += 1.0f;
	val[1] += 1.0f;
	
	//printf("val = %f\n", *val);
	release_data(&my_float_state, node, 0);
}

void cublas_codelet(void *_args)
{
	unsigned node;
	node = get_local_memory_node();
	//printf("codelet cublas on node %d\n", node);

	float *val;
	val = (float *)fetch_data(&my_float_state, node, 1, 1);
	dunity = (float *)fetch_data(&unity_state, node, 1, 0);
//	*val = *val +1;
//	printf("BEFORE %f \n", my_lovely_float);
	cublasSaxpy(3, 1.0f, dunity, 1, val, 1);

	/* write-through is needed here ! */
	release_data(&my_float_state, node, 1<<0);
//	printf("AFTER %f \n", my_lovely_float);
}

int main(int argc, char **argv)
{
	init_machine();
	init_workers();

	monitor_new_data(&my_float_state, 0 /* home node */,
		(uintptr_t)&my_lovely_float, sizeof(my_lovely_float));

	monitor_new_data(&unity_state, 0 /* home node */,
		(uintptr_t)&unity, sizeof(unity));

//	init_memory_nodes();

	printf("ok\n");

	codelet cl;

	job_t j;
 
	/* ok that is dummy
	 */ 

	unsigned i;
	for (i = 0; i < NITER; i++)
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

	return 0;
}
