#include <core/policies/eager-central-policy.h>

/*
 *	This is just the trivial policy where every worker use the same
 *	JOB QUEUE.
 */

/* the former is the actual queue, the latter some container */
static struct jobq_s jobq;


static void set_worker_queues(struct machine_config_s *config)
{
#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < config->ncores; core++)
	{
		core_worker_arg *corearg = &config->coreargs[core];
		corearg->jobq = &jobq;
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < config->ncudagpus; cudadev++)
	{
		cuda_worker_arg *cudaarg = &config->cudaargs[cudadev];
		cudaarg->jobq = &jobq;
	}
#endif

#ifdef USE_CUBLAS
	/* initialize CUBLAS with the proper number of threads */
	unsigned cublasdev;
	for (cublasdev = 0; cublasdev < config->ncublasgpus; cublasdev++)
	{
		cublas_worker_arg *cublasarg = &config->cublasargs[cublasdev]; 
		cublasarg->jobq = &jobq;
	}
#endif

#ifdef USE_SPU
	/* initialize the various SPUs  */
	unsigned spu;
	for (spu = 0; spu < config->nspus; spu++)
	{
		spu_worker_arg *spuarg = &config->spuargs[spu];

		spuarg->jobq = &jobq;
	}
#endif
}

void initialize_eager_center_policy(struct machine_config_s *config, 
			__attribute__ ((unused))	struct sched_policy_s *_policy) 
{
	init_central_jobq(&jobq);

	set_worker_queues(config);

	jobq.push_task = central_push_task;
	jobq.push_prio_task = central_push_prio_task;
	jobq.pop_task = central_pop_task;
}

struct jobq_s *get_local_queue_eager(struct sched_policy_s *policy __attribute__ ((unused)))
{
	/* this is trivial for that strategy :) */

	//struct jobq_s *queue;
	//queue = pthread_getspecific(policy->local_queue_key);
	//printf("get local queue -> %p\n", queue);

	return &jobq;
}


