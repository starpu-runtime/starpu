#include "queues.h"

/*
 * There can be various queue designs
 * 	- trivial single list
 * 	- cilk-like 
 * 	- hierarchical (marcel-like)
 */

void setup_queues(void (*init_queue_design)(void),
		  struct jobq_s *(*func_init_queue)(void), 
		  struct machine_config_s *config) 
{
	init_queue_design();

#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < config->ncores; core++)
	{
		core_worker_arg *corearg = &config->coreargs[core];
		corearg->jobq = func_init_queue();
		corearg->jobq->who |= CORE;
		corearg->jobq->alpha = CORE_ALPHA;
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < config->ncudagpus; cudadev++)
	{
		cuda_worker_arg *cudaarg = &config->cudaargs[cudadev];
		cudaarg->jobq = func_init_queue();
		cudaarg->jobq->who |= CUBLAS|CUDA;
		cudaarg->jobq->alpha = CUDA_ALPHA;
	}
#endif

#ifdef USE_CUBLAS
	/* initialize CUBLAS with the proper number of threads */
	unsigned cublasdev;
	for (cublasdev = 0; cublasdev < config->ncublasgpus; cublasdev++)
	{
		cublas_worker_arg *cublasarg = &config->cublasargs[cublasdev]; 
		cublasarg->jobq = func_init_queue();
		cublasrg->jobq->who |= CUBLAS;
		cublasarg->jobq->alpha = CUBLAS_ALPHA;
	}
#endif

#ifdef USE_SPU
	/* initialize the various SPUs  */
	unsigned spu;
	for (spu = 0; spu < config->nspus; spu++)
	{
		spu_worker_arg *spuarg = &config->spuargs[spu];

		spuarg->jobq = func_init_queue();

		spuarg->jobq->who |= SPU;
	}
#endif
}

/* this may return NULL for an "anonymous thread" */
struct jobq_s *get_local_queue(void)
{
	struct sched_policy_s *policy = get_sched_policy();

	return pthread_getspecific(policy->local_queue_key);
}

/* XXX how to retrieve policy ? that may be given in the machine config ? */
void set_local_queue(struct jobq_s *jobq)
{
	struct sched_policy_s *policy = get_sched_policy();

	pthread_setspecific(policy->local_queue_key, jobq);
}
