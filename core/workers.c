#include <core/workers.h>

static struct machine_config_s config;

/* in case a task is submitted, we may check whether there exists a worker
   that may execute the task or not */
static uint32_t worker_mask = 0;

inline uint32_t worker_exists(uint32_t task_mask)
{
	return (task_mask & worker_mask);
} 

inline uint32_t may_submit_cuda_task(void)
{
	return ((CUDA|CUBLAS) & worker_mask);
}

inline uint32_t may_submit_core_task(void)
{
	return (CORE & worker_mask);
}

static unsigned ncores;
static unsigned ncudagpus;
static unsigned ngordon_spus;

/*
 * Runtime initialization methods
 */

#ifdef USE_CUDA
extern unsigned get_cuda_device_count(void);
#endif

static void init_machine_config(struct machine_config_s *config)
{
	int envval;

	config->nworkers = 0;
	
#ifdef USE_CPUS
	envval = get_env_number("NCPUS");
	if (envval < 0) {
		ncores = MIN(sysconf(_SC_NPROCESSORS_ONLN), NMAXCORES);
	} else {
		/* use the specified value */
		ncores = (unsigned)envval;
		STARPU_ASSERT(ncores <= NMAXCORES);
	}
	STARPU_ASSERT(ncores + config->nworkers <= NMAXWORKERS);

	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		config->workers[config->nworkers + core].arch = CORE_WORKER;
		config->workers[config->nworkers + core].perf_arch = CORE_DEFAULT;
		config->workers[config->nworkers + core].id = core;
		worker_mask |= CORE;
	}

	config->nworkers += ncores;
#endif

#ifdef USE_CUDA
	/* we need to initialize CUDA early to count the number of devices */
	init_cuda();

	envval = get_env_number("NCUDA");
	if (envval < 0) {
		ncudagpus = MIN(get_cuda_device_count(), MAXCUDADEVS);
	} else {
		/* use the specified value */
		ncudagpus = (unsigned)envval;
		STARPU_ASSERT(ncudagpus <= MAXCUDADEVS);
	}
	STARPU_ASSERT(ncudagpus + config->nworkers <= NMAXWORKERS);

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < ncudagpus; cudagpu++)
	{
		config->workers[config->nworkers + cudagpu].arch = CUDA_WORKER;
		config->workers[config->nworkers + cudagpu].perf_arch = CUDA_DEFAULT;
		config->workers[config->nworkers + cudagpu].id = cudagpu;
		worker_mask |= (CUDA|CUBLAS);
	}

	config->nworkers += ncudagpus;
#endif
	
#ifdef USE_GORDON
	envval = get_env_number("NGORDON");
	if (envval < 0) {
		ngordon_spus = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
	} else {
		/* use the specified value */
		ngordon_spus = (unsigned)envval;
		STARPU_ASSERT(ngordon_spus <= NMAXGORDONSPUS);
	}
	STARPU_ASSERT(ngordon_spus + config->nworkers <= NMAXWORKERS);

	unsigned spu;
	for (spu = 0; spu < ngordon_spus; spu++)
	{
		config->workers[config->nworkers + spu].arch = GORDON_WORKER;
		config->workers[config->nworkers + spu].perf_arch = GORDON_DEFAULT;
		config->workers[config->nworkers + spu].id = spu;
		config->workers[config->nworkers + spu].worker_is_running = 0;
		worker_mask |= GORDON;
	}

	config->nworkers += ngordon_spus;
#endif

}

static void init_workers_binding(struct machine_config_s *config)
{
	/* launch one thread per CPU */
	unsigned ram_memory_node;

	int current_bindid = 0;

	/* note that even if the CPU core are not used, we always have a RAM node */
	/* TODO : support NUMA  ;) */
	ram_memory_node = register_memory_node(RAM);

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		unsigned memory_node;
		struct worker_s *workerarg = &config->workers[worker];
		
		/* "dedicate" a cpu core to that worker */
		workerarg->bindid =
			(current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		/* select the memory node that contains worker's memory */
		switch (workerarg->arch) {
			case CORE_WORKER:
			case GORDON_WORKER:
				memory_node = ram_memory_node;
				break;
			case CUDA_WORKER:
				memory_node = register_memory_node(CUDA_RAM);
				break;
			default:
				STARPU_ASSERT(0);
		}

		workerarg->memory_node = memory_node;
	}
}

#ifdef USE_GORDON
unsigned gordon_inited = 0;	
struct worker_set_s gordon_worker_set;
#endif

static void init_workers(struct machine_config_s *config)
{
	config->running = 1;

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct worker_s *workerarg = &config->workers[worker];

		sem_init(&workerarg->ready_sem, 0, 0);

		/* if some codelet's termination cannot be handled directly :
		 * for instance in the Gordon driver, Gordon tasks' callbacks
		 * may be executed by another thread than that of the Gordon
		 * driver so that we cannot call the push_codelet_output method
		 * directly */
		workerarg->terminated_jobs = job_list_new();
	
		switch (workerarg->arch) {
			case CORE_WORKER:
				workerarg->set = NULL;
				pthread_create(&workerarg->worker_thread, 
						NULL, core_worker, workerarg);
				sem_wait(&workerarg->ready_sem);
				break;
#ifdef USE_CUDA
			case CUDA_WORKER:
				workerarg->set = NULL;
				pthread_create(&workerarg->worker_thread, 
						NULL, cuda_worker, workerarg);
				sem_wait(&workerarg->ready_sem);
				break;
#endif
#ifdef USE_GORDON
			case GORDON_WORKER:
				/* we will only launch gordon once, but it will handle 
				 * the different SPU workers */
				if (!gordon_inited)
				{
					gordon_worker_set.nworkers = ngordon_spus; 
					gordon_worker_set.workers = &config->workers[worker];

					pthread_create(&gordon_worker_set.worker_thread, NULL, 
							gordon_worker, &gordon_worker_set);
					sem_wait(&gordon_worker_set.ready_sem);

					gordon_inited = 1;
				}
				
				workerarg->set = &gordon_worker_set;
				gordon_worker_set.joined = 0;
				workerarg->worker_is_running = 1;

				break;
#endif
			default:
				STARPU_ASSERT(0);
		}
	}
}

void init_machine(void)
{
	srand(2008);

#ifdef USE_FXT
	start_fxt_profiling();
#endif

	timing_init();

	init_machine_config(&config);

	/* for the data wizard */
	init_memory_nodes();

	init_workers_binding(&config);

	/* initialize the scheduler */

	/* initialize the queue containing the jobs */
	init_sched_policy(&config);

	init_workers(&config);
}

/*
 * Handle runtime termination 
 */

void terminate_workers(struct machine_config_s *config)
{
	fprintf(stderr, "terminate workers \n");

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		void *retval;
		wake_all_blocked_workers();
		
		fprintf(stderr, "wait for worker %d\n", worker);

		if (config->workers[worker].set){ 
			if (!config->workers[worker].set->joined) {
				pthread_join(config->workers[worker].set->worker_thread, &retval);
				config->workers[worker].set->joined = 1;
				if (retval)
					fprintf(stderr, "(set) worker %d returned %p!\n", worker, retval);
			}
		}
		else {
			pthread_join(config->workers[worker].worker_thread, &retval);
			if (retval)
				fprintf(stderr, "worker %d returned %p!\n", worker, retval);
		}
	}
}

unsigned machine_is_running(void)
{
	return config.running;
}

void kill_all_workers(struct machine_config_s *config)
{
	/* set the flag which will tell workers to stop */
	config->running = 0;

	/* in case some workers are waiting on some event 
	   wake them up ... */
	wake_all_blocked_workers();
}

void terminate_machine(void)
{
	display_msi_stats();

	/* tell all workers to shutdown */
	kill_all_workers(&config);

	if (get_env_number("CALIBRATE") != -1)
		dump_registered_models();

	/* wait for their termination */
	terminate_workers(&config);
}
