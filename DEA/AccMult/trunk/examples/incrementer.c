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
#include <datawizard/hierarchy.h>
#include <datawizard/filters.h>

#ifdef USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef USE_GORDON
#include "../drivers/gordon/externals/scalp/cell/gordon/gordon.h"
#endif

#define NITER	10000

data_state *my_float_state;
data_state *my_float_state2;
data_state my_float_root_state;
data_state unity_state;
data_state pouet_state;

//float my_lovely_float[3] = {0.0f, 0.0f, 0.0f};
float my_lovely_float[6] = {0.0f, 0.0f, 0.0f,
			    0.0f, 0.0f, 0.0f};
float unity[3] = {1.0f, 0.0f, 1.0f};


void cuda_callback_func(__attribute__ ((unused)) void *argcb)
{
	printf("CUDA codelet done\n");
}

void callback_func(__attribute__ ((unused)) void *argcb)
{
	int cntleft = (int)my_lovely_float[0];
	int cntright = (int)my_lovely_float[3];

	if ((cntleft == NITER) && (cntright == NITER)) 
	{
		printf("LEFT -> %f, %f, %f \n", my_lovely_float[0],
				my_lovely_float[1], my_lovely_float[2]);
		printf("RIGHT -> %f, %f, %f \n", my_lovely_float[3], 
				my_lovely_float[4], my_lovely_float[5]);
		printf("stopping ...\n");
		unpartition_data(&my_float_root_state, 0);
		exit(0);
	}

}

void core_codelet(buffer_descr *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].ptr;

	val[0] += 1.0f; val[1] += 1.0f;
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void cublas_codelet(buffer_descr *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].ptr;
	float *dunity = (float *)buffers[1].ptr;

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
#endif

#ifdef USE_CUDA
struct cuda_module_s cuda_module;
struct cuda_function_s cuda_function;

void initialize_cuda(void)
{
	char *module_path = 
		"/home/gonnet/DEA/AccMult/examples/cuda/incrementer_cuda.cubin";
	char *function_symbol = "cuda_incrementer";

	init_cuda_module(&cuda_module, module_path);
	init_cuda_function(&cuda_function, &cuda_module, function_symbol);

}

cuda_codelet_t arg_cuda;

static int foo[2];

void launch_cuda_codelet(void)
{
	job_t j;
	tag_t tag;

	codelet cl;

	arg_cuda.func = &cuda_function;
	arg_cuda.stack = &foo[0];
	arg_cuda.stack_size = 2*sizeof(int); 

	arg_cuda.gridx = 1;
	arg_cuda.gridy = 1;

	arg_cuda.blockx = 1;
	arg_cuda.blocky = 1;

	arg_cuda.shmemsize = 1024;

	cl.cuda_func = &arg_cuda;
	cl.cl_arg = NULL;

	j = job_new();
	j->type = CODELET;
	j->where = CUDA;
	j->cb = cuda_callback_func;
	j->argcb = NULL;
	j->cl = &cl;

	j->nbuffers = 0;

	tag =	((1664ULL)<<32);
	tag_declare(tag, j);

	push_task(j);

}
#endif

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned i;
	tag_t tag;

	init_machine();
	init_workers();

	monitor_new_data(&my_float_root_state, 0 /* home node */,
		(uintptr_t)&my_lovely_float, 6, 6, 1, sizeof(float));

	monitor_new_data(&unity_state, 0 /* home node */,
		(uintptr_t)&unity, 3, 3, 1, sizeof(float));

	filter f;
		f.filter_func = block_filter_func;
		f.filter_arg = 2;

	partition_data(&my_float_root_state, &f);

#ifdef USE_CUDA
	initialize_cuda();
#endif

	my_float_state = &my_float_root_state.children[0];
	my_float_state2 = &my_float_root_state.children[1];

	codelet cl;
	codelet cl2;
	job_t j;

#ifdef USE_GORDON
	codelet cl_gordon;

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


	cl.cl_arg = my_float_state;
	cl.core_func = core_codelet;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	cl.cublas_func = cublas_codelet;
#endif

	cl2.cl_arg = my_float_state2;
	cl2.core_func = core_codelet;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	cl2.cublas_func = cublas_codelet;
#endif

#ifdef USE_CUDA
	launch_cuda_codelet();
#endif

	for (i = 0; i < NITER; i++)
	{

		j = job_new();
		j->type = CODELET;
		j->where = ANY;
		j->cb = callback_func;
		j->argcb = NULL;
		j->cl = &cl;

		j->nbuffers = 2;
		j->buffers[0].state = my_float_state; 
		j->buffers[0].mode = RW;
		j->buffers[1].state = &unity_state; 
		j->buffers[1].mode = R;

		tag =	((1ULL)<<32 | (unsigned long long)(i));
		tag_declare(tag, j);

		push_task(j);


		j = job_new();
		j->type = CODELET;
		j->where = CORE;
		j->cb = callback_func;
		j->argcb = NULL;
		j->cl = &cl2;

		j->nbuffers = 2;
		j->buffers[0].state = my_float_state2;
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
