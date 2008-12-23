#include <core/workers.h>

static struct machine_config_s config;

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
	unsigned ncores;
	envval = get_env_number("NCPUS");
	if (envval < 0) {
		ncores = MIN(sysconf(_SC_NPROCESSORS_ONLN), NMAXCORES);
	} else {
		/* use the specified value */
		ncores = (unsigned)envval;
		ASSERT(ncores <= NMAXCORES);
	}
	ASSERT(ncores + config->nworkers <= NMAXWORKERS);

	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		config->workers[config->nworkers + core].arch = CORE_WORKER;
		config->workers[config->nworkers + core].perf_arch = CORE_DEFAULT;
		config->workers[config->nworkers + core].id = core;
	}

	config->nworkers += ncores;

#endif

#ifdef USE_CUDA
	/* we need to initialize CUDA early to count the number of devices */
	unsigned ncudagpus;
	init_cuda();

	envval = get_env_number("NCUDA");
	if (envval < 0) {
		ncudagpus = MIN(get_cuda_device_count(), MAXCUDADEVS);
	} else {
		/* use the specified value */
		ncudagpus = (unsigned)envval;
		ASSERT(ncudagpus <= MAXCUDADEVS);
	}
	ASSERT(ncudagpus + config->nworkers <= NMAXWORKERS);

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < ncudagpus; cudagpu++)
	{
		config->workers[config->nworkers + cudagpu].arch = CUDA_WORKER;
		config->workers[config->nworkers + cudagpu].perf_arch = CUDA_DEFAULT;
		config->workers[config->nworkers + cudagpu].id = cudagpu;
	}

	config->nworkers += ncudagpus;
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
				memory_node = ram_memory_node;
				break;
			case CUDA_WORKER:
				memory_node = register_memory_node(CUDA_RAM);
				break;
			default:
				ASSERT(0);
		}

		workerarg->memory_node = memory_node;
	}
}



static void init_workers(struct machine_config_s *config)
{
	config->running = 1;

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct worker_s *workerarg = &config->workers[worker];

		sem_init(&workerarg->ready_sem, 0, 0);
	
		void *(*worker_func)(void *);	
		switch (workerarg->arch) {
			case CORE_WORKER:
				worker_func = core_worker;
				break;
#ifdef USE_CUDA
			case CUDA_WORKER:
				worker_func = cuda_worker;
				break;
#endif
			default:
				ASSERT(0);
		}

		thread_create(&workerarg->worker_thread, 
					NULL, worker_func, workerarg);

		sem_wait(&workerarg->ready_sem);
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
		thread_join(config->workers[worker].worker_thread, &retval);
		fprintf(stderr, "worker %d returned %p!\n", worker, retval);
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
