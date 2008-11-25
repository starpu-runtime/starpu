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
#include <datawizard/datawizard.h>

#ifdef USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef USE_GORDON
#include "../drivers/gordon/externals/scalp/cell/gordon/gordon.h"
#endif

#define NITER	100000

data_state my_float_state;
data_state unity_state;

float my_lovely_float[5] = {0.0f, 0.0f, 0.0f, 1664.0f, 1664.0f};
float unity[5] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f};


void callback_func(void *argcb)
{
	unsigned cnt = ATOMIC_ADD((unsigned *)argcb, 1);

//	printf("cnt %d vs. NITER %d\n", *cnt, NITER);

	if ((cnt == NITER)) 
	{
		/* stop monitoring data and grab it in RAM */
		release_data(&my_float_state, 1<<0);

		printf("delete data ...\n");
		delete_data(&my_float_state);
		
		printf("RIGHT -> %f, %f, %f\n", my_lovely_float[0], 
				my_lovely_float[1], my_lovely_float[2]);
		printf("stopping ...\n");

		terminate_machine();
	
		exit(0);
	}

}

void core_codelet(data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].vector.ptr;

	val[0] += 1.0f; val[1] += 1.0f;
}

#ifdef USE_CUDA
void cublas_codelet(data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].vector.ptr;
	float *dunity = (float *)buffers[1].vector.ptr;

	cublasSaxpy(3, 1.0f, dunity, 1, val, 1);
}
#endif

#ifdef USE_GORDON
#define BUFFER_SIZE	32

void gordon_callback_func(void *argcb)
{
	printf("gordon_callback_func\n");
	/* this is not used yet ! XXX  */
}

void gordon_codelet(__attribute__ ((unused)) void *_args)
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

void gordon_test(void)
{
	codelet cl_gordon;

	j = job_create();
	j->where = GORDON;
	j->cb = gordon_callback_func;
	j->cl = &cl_gordon;

	cl_gordon.gordon_func = gordon_codelet;
	cl_gordon.cl_arg = NULL;

	push_task(j);
}

#endif

#ifdef USE_CUDA
static struct cuda_module_s cuda_module;
static struct cuda_function_s cuda_function;

static cuda_codelet_t cuda_codelet;

void initialize_cuda(void)
{
	char *module_path = 
		"/home/gonnet/DEA/AccMult/examples/cuda/incrementer_cuda.cubin";
	char *function_symbol = "cuda_incrementer";

	init_cuda_module(&cuda_module, module_path);
	init_cuda_function(&cuda_function, &cuda_module, function_symbol);

	cuda_codelet.func = &cuda_function;
	cuda_codelet.stack = NULL;
	cuda_codelet.stack_size = 0; 

	cuda_codelet.gridx = 1;
	cuda_codelet.gridy = 1;

	cuda_codelet.blockx = 1;
	cuda_codelet.blocky = 1;

	cuda_codelet.shmemsize = 1024;
}

#endif

void init_data(void)
{
	monitor_vector_data(&my_float_state, 0 /* home node */, (uintptr_t)&my_lovely_float, 5, sizeof(float));

	monitor_vector_data(&unity_state, 0 /* home node */, (uintptr_t)&unity, 5, sizeof(float));
}

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned i;
	unsigned counter;

	tag_t tag;

	init_machine();

	init_data();

#ifdef USE_CUDA
	initialize_cuda();
#endif

	codelet cl;
	job_t j;

	counter = 0;

	cl.cl_arg = &counter;
	cl.core_func = core_codelet;
#ifdef USE_CUDA
	//cl.cublas_func = cublas_codelet;
	cl.cuda_func = &cuda_codelet;
#endif

	for (i = 0; i < NITER; i++)
	{
		j = job_create();
#ifdef USE_CUDA
		j->where = CUDA|CORE;
#endif
			//(((i % 2) == 1)?CUDA:CUBLAS)|CORE; 
		
		j->cb = callback_func;
		j->cl = &cl;
		j->argcb = &counter;

		j->nbuffers = 2;
		j->buffers[0].state = &my_float_state;
		j->buffers[0].mode = RW;
		j->buffers[1].state = &unity_state; 
		j->buffers[1].mode = R;

		tag =	((2ULL)<<32 | (unsigned long long)(i));
		tag_declare(tag, j);

		push_task(j);
	}

	sleep(100);

	return 0;
}
