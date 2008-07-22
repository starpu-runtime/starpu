#include <core/policies/work-stealing-policy.h>

/* save the general machine configuration */
static struct machine_config_s *machineconfig;

/* XXX 16 is set randomly */
unsigned nworkers;
unsigned rr_worker;
struct jobq_s *queue_array[16];

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
//static unsigned total_jobs;

/* who to steal work to ? */
struct jobq_s *select_victimq(void)
{
	struct jobq_s *q;

	q = queue_array[rr_worker];

	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

/* when anonymous threads submit tasks, 
 * we need to select a queue where to dispose them */
struct jobq_s *select_workerq(void)
{
	struct jobq_s *q;

	q = queue_array[rr_worker];

	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

static job_t ws_pop_task(struct jobq_s *q)
{
	job_t j;

	j = ws_non_blocking_pop_task(q);
	if (j) {
		/* there was a local task */
		return j;
	}
	
	/* we need to steal someone's job */
	do {
		struct jobq_s *victimq;
		victimq = select_victimq();

		j = ws_non_blocking_pop_task_if_job_exists(victimq);

	} while(!j);

	TRACE_WORK_STEALING(q, j);

	return j;
}

static struct jobq_s *init_ws_deque(void)
{
	struct jobq_s *q;

	q = create_deque();

	q->push_task = ws_push_task; 
	q->push_prio_task = ws_push_prio_task; 
	q->pop_task = ws_pop_task;
	q->who = 0;

	queue_array[nworkers++] = q;

	return q;
}

static void set_worker_ws_queues(struct machine_config_s *config)
{
#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < config->ncores; core++)
	{
		core_worker_arg *corearg = &config->coreargs[core];
		corearg->jobq = init_ws_deque();
		corearg->jobq->who = CORE;
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < config->ncudagpus; cudadev++)
	{
		cuda_worker_arg *cudaarg = &config->cudaargs[cudadev];
		cudaarg->jobq = init_ws_deque();
		cudaarg->jobq->who = CUBLAS|CUDA;
	}
#endif

#ifdef USE_CUBLAS
	/* initialize CUBLAS with the proper number of threads */
	unsigned cublasdev;
	for (cublasdev = 0; cublasdev < config->ncublasgpus; cublasdev++)
	{
		cublas_worker_arg *cublasarg = &config->cublasargs[cublasdev]; 
		cublasarg->jobq = init_ws_deque();
		cublasrg->jobq->who = CUBLAS;
	}
#endif

#ifdef USE_SPU
	/* initialize the various SPUs  */
	unsigned spu;
	for (spu = 0; spu < config->nspus; spu++)
	{
		spu_worker_arg *spuarg = &config->spuargs[spu];

		spuarg->jobq = init_ws_deque();
		spuarg->jobq->who = SPU;
	}
#endif
}

void initialize_ws_policy(struct machine_config_s *config, 
			__attribute__ ((unused))	struct sched_policy_s *_policy) 
{
	nworkers = 0;
	rr_worker = 0;

	machineconfig = config;

	init_ws_queues_mechanisms();

	set_worker_ws_queues(config);
}

struct jobq_s *get_local_queue_ws(struct sched_policy_s *policy __attribute__ ((unused)))
{
	struct jobq_s *queue;
	queue = pthread_getspecific(policy->local_queue_key);

	if (!queue) {
		queue = select_workerq();
	}

	ASSERT(queue);

	return queue;
}

