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

#define NITER	1000

data_state my_float_state;
data_state unity_state;

sem_t sem;

unsigned size __attribute__ ((aligned (16))) = 4*sizeof(float);

float my_lovely_float[4] __attribute__ ((aligned (16))) = { 0.0f, 0.0f, 0.0f, 1664.0f}; 
float unity[4] __attribute__ ((aligned (16))) = { 1.0f, 0.0f, 1.0f, 0.0f };

void callback_func(void *argcb)
{
	unsigned cnt = ATOMIC_ADD((unsigned *)argcb, 1);

	if ((cnt == NITER)) 
	{
		/* stop monitoring data and grab it in RAM */
		unpartition_data(&my_float_state, 0);

		printf("delete data ...\n");
		delete_data(&my_float_state);
		
		printf("array -> %f, %f, %f\n", my_lovely_float[0], 
				my_lovely_float[1], my_lovely_float[2]);
		printf("stopping ...\n");

		terminate_machine();

		sem_post(&sem);
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

#ifdef USE_CUDA
static struct cuda_module_s cuda_module;
static struct cuda_function_s cuda_function;

static cuda_codelet_t cuda_codelet;

void initialize_cuda(void)
{
	char module_path[1024];
	sprintf(module_path, 
		"%s/examples/cuda/incrementer_cuda.cubin", STARPUDIR);
	char *function_symbol = "cuda_incrementer";

	init_cuda_module(&cuda_module, module_path);
	init_cuda_function(&cuda_function, &cuda_module, function_symbol);

	cuda_codelet.func = &cuda_function;

	cuda_codelet.gridx = 1;
	cuda_codelet.gridy = 1;

	cuda_codelet.blockx = 1;
	cuda_codelet.blocky = 1;

	cuda_codelet.shmemsize = 1024;
}
#endif

void init_data(void)
{
	monitor_vector_data(&my_float_state, 0 /* home node */, (uintptr_t)&my_lovely_float, 4, sizeof(float));

	monitor_vector_data(&unity_state, 0 /* home node */, (uintptr_t)&unity, 4, sizeof(float));
}

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned i;
	unsigned counter;

	tag_t tag;

	init_machine();
	fprintf(stderr, "StarPU initialized ...\n");

	sem_init(&sem, 0, 0);

	init_data();

#ifdef USE_CUDA
	initialize_cuda();
#endif

	codelet cl;
	job_t j;

	counter = 0;

	cl.cl_arg = &size;
	cl.cl_arg_size = sizeof(unsigned);

	cl.core_func = core_codelet;
#ifdef USE_CUDA
	//cl.cublas_func = cublas_codelet;
	cl.cuda_func = &cuda_codelet;
#endif


#ifdef USE_GORDON
	cl.gordon_func = SPU_FUNC_ADD;
#endif


	for (i = 0; i < NITER; i++)
	{
		j = job_create();
		j->where = CORE;
#ifdef USE_CUDA
		j->where |= CUDA;
#endif
#ifdef USE_GORDON
		j->where |= GORDON;
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

	sem_wait(&sem);

	return 0;
}
