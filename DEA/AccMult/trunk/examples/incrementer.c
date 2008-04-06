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

#ifdef USE_GORDON
#include "../drivers/gordon/externals/scalp/cell/gordon/gordon.h"
#endif

#define NITER	1000

data_state my_float_state;
data_state unity_state;
data_state my_foo_state;

float my_lovely_float[3] = {0.0f, 0.0f, 0.0f};
float unity[3] = {1.0f, 0.0f, 1.0f};
float *dunity;

void callback_func(void *argcb)
{
	if ((int)my_lovely_float[0] % 1000000 == 0) {
		printf("-> %f, %f, %f \n", my_lovely_float[0], my_lovely_float[1], my_lovely_float[2]);
	}

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
	float *val;

	node = get_local_memory_node();

	val = (float *)fetch_data(&my_float_state, node, 1, 1);
	val[0] += 1.0f; val[1] += 1.0f;
	
	release_data(&my_float_state, node, 0);
}

#ifdef USE_CUBLAS
void cublas_codelet(void *_args)
{
	unsigned node;
	float *val;
	node = get_local_memory_node();

	val = (float *)fetch_data(&my_float_state, node, 1, 1);
	dunity = (float *)fetch_data(&unity_state, node, 1, 0);
	cublasSaxpy(3, 1.0f, dunity, 1, val, 1);

	/* write-through is needed here ! */
	release_data(&my_float_state, node, 1<<0);
}
#endif

#ifdef USE_GORDON

#define BUFFER_SIZE	32

void gordon_callback_func(void *argcb)
{
	printf("gordon_callback_func\n");
	/* this is not used yet ! XXX  */
}

void gordon_codelet(void *_args)
{
	printf("gordon codelet\n");
	struct gordon_ppu_job_s *joblist = gordon_alloc_jobs(2, 0);
	float *array = gordon_malloc(BUFFER_SIZE);
	float *output = gordon_malloc(BUFFER_SIZE);
	int i = 0, n;

	int *nptr = gordon_malloc(sizeof(int));
	n = *nptr = BUFFER_SIZE / sizeof(float);

	for (i = 0; i < n; i++) {
		array[i] = (float)i;
	}
	
	joblist[0].index  = FUNC_A;
	joblist[0].nalloc = 0;
	
	joblist[0].nin    = 0;
	joblist[0].ninout = 0;
	joblist[0].nout   = 0;
	
	joblist[1].index  = FUNC_B;
	joblist[1].nalloc = 0;
	joblist[1].nin    = 2;
	joblist[1].ninout = 0;
	joblist[1].nout   = 1;
	
	joblist[1].buffers[0] = (uint64_t)nptr;
	joblist[1].ss[0].size = sizeof(int);
	joblist[1].buffers[1] = (uint64_t)array;
	joblist[1].ss[1].size = BUFFER_SIZE;
	joblist[1].buffers[2] = (uint64_t)output;
	joblist[1].ss[2].size = BUFFER_SIZE;

	gordon_pushjob(&joblist[0], gordon_callback_func, output);

	gordon_join();
}

#endif

int main(int argc, char **argv)
{
	init_machine();
	printf("machine inited \n");
	init_workers();
	printf("workers inited \n");

	uint32_t fooooo;
	monitor_new_data(&my_foo_state, 0 /* home node */,
		(uintptr_t)&fooooo, sizeof(uint32_t));

	monitor_new_data(&my_float_state, 0 /* home node */,
		(uintptr_t)&my_lovely_float, sizeof(my_lovely_float));

	monitor_new_data(&unity_state, 0 /* home node */,
		(uintptr_t)&unity, sizeof(unity));

	codelet cl;
	codelet cl_gordon;
	job_t j;

#ifdef USE_GORDON
	j = job_new();
	j->type = CODELET;
	j->where = GORDON;
	j->cb = gordon_callback_func;
	j->argcb = NULL;
	j->cl = &cl_gordon;

	cl_gordon.gordon_func = gordon_codelet;
	cl_gordon.cl_arg = NULL;

	push_task(j);
#endif

	unsigned i;
	for (i = 0; i < NITER; i++)
	{
		j = job_new();
		j->type = CODELET;
		j->where = CORE;
		j->cb = callback_func;
		j->argcb = NULL;
		j->cl = &cl;

		cl.cl_arg = NULL;
		cl.core_func = core_codelet;
#ifdef USE_CUBLAS
		cl.cublas_func = cublas_codelet;
#endif
		push_task(j);
	}

	sleep(100);

	return 0;
}
