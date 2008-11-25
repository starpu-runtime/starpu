#include "workers.h"

static struct machine_config_s config;

/*
 * Runtime initialization methods
 */

static void init_machine_config(struct machine_config_s *config)
{
	int envval;
	
#ifdef USE_CPUS
	envval = get_env_number("NCPUS");
	if (envval < 0) {
		config->ncores = MIN(sysconf(_SC_NPROCESSORS_ONLN), NMAXCORES);
	} else {
		/* use the specified value */
		config->ncores = (unsigned)envval;
		ASSERT(config->ncores <= NMAXCORES);
	}
#endif

#ifdef USE_CUDA
	/* we to initialize CUDA early to count the number of devices */
	init_cuda();

	envval = get_env_number("NCUDA");
	if (envval < 0) {
		config->ncudagpus = MIN(get_cuda_device_count(), MAXCUDADEVS);
	} else {
		/* use the specified value */
		config->ncudagpus = (unsigned)envval;
		ASSERT(config->ncudagpus <= MAXCUDADEVS);
	}

#endif

#ifdef USE_SPU
	envval = get_env_number("NSPUS");
	if (envval < 0) {
		config->nspus = MIN(get_spu_count(), MAXSPUS);
	} else {
		/* use the specified value */
		config->nspus = (unsigned)envval;
		ASSERT(config->nspus <= MAXSPUS);
	}

#endif
}

static void init_workers_binding(struct machine_config_s *config)
{
	/* launch one thread per CPU */
	unsigned memory_node;

	int current_bindid = 0;

	/* note that even if the CPU core are not used, we always have a RAM node */
	/* TODO : support NUMA  ;) */
	memory_node = register_memory_node(RAM);

#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < config->ncores; core++)
	{
		core_worker_arg *corearg = &config->coreargs[core];

		corearg->bindid =
			(current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		corearg->memory_node = memory_node;
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < config->ncudagpus; cudadev++)
	{
		cuda_worker_arg *cudaarg = &config->cudaargs[cudadev];

		cudaarg->bindid =
			(current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		cudaarg->memory_node = register_memory_node(CUDA_RAM);
	}
#endif

#ifdef USE_SPU
	/* initialize the various SPUs  */
	unsigned spu;
	for (spu = 0; spu < config->nspus; spu++)
	{
		spu_worker_arg *spuarg = &config->spuargs[spu];

		spuarg->bindid =
			(current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		spuarg->memory_node = register_memory_node(SPU_LS);
	}
#endif

#ifdef USE_GORDON
	config->gordonargs.bindid = 
		(current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));
	/* XXX do not forget to registrate memory nodes for each SPUs later on ! */
#endif
}



static void init_workers(struct machine_config_s *config)
{
	config->running = 1;

#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < config->ncores; core++)
	{
		core_worker_arg *corearg = &config->coreargs[core];

		corearg->coreid = core;
		corearg->ready_flag = 0;

		thread_create(&config->corethreads[core], 
					NULL, core_worker, corearg);

		/* wait until the thread is actually launched ... */
		while (corearg->ready_flag == 0) {}
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < config->ncudagpus; cudadev++)
	{
		cuda_worker_arg *cudaarg = &config->cudaargs[cudadev];

		cudaarg->deviceid = cudadev;
		cudaarg->ready_flag = 0;

		thread_create(&config->cudathreads[cudadev], 
				NULL, cuda_worker, (void*)cudaarg);

		/* wait until the thread is actually launched ... */
		while (cudaarg->ready_flag == 0) {}
	}
#endif

#ifdef USE_SPU
	/* initialize the various SPUs  */
	unsigned spu;
	for (spu = 0; spu < config->nspus; spu++)
	{
		spu_worker_arg *spuarg = &config->spuargs[spu];

		spuarg->deviceid = spu;
		spuarg->ready_flag = 0;

		thread_create(&config->sputhreads[spu], 
			NULL, spu_worker, (void*)spuarg);

		/* wait until the thread is actually launched ... */
		while (spuarg->ready_flag == 0) {}
	}
#endif

#ifdef USE_GORDON
	config->ngordonspus = 8;
	config->gordonargs.ready_flag = 0;

	config->gordonargs.nspus = ngordonspus;

	thread_create(&gordonthread, NULL, 
			gordon_worker, (void*)&config->gordonargs);

	/* wait until the thread is actually launched ... */
	while (config->gordonargs.ready_flag == 0) {}
#endif
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

#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < config->ncores; core++)
	{
		thread_join(config->corethreads[core], NULL);
	}
	fprintf(stderr, "core terminated\n");
#endif

#ifdef USE_CUDA
	int cudadev;
	for (cudadev = 0; cudadev < config->ncudagpus; cudadev++)
	{
		thread_join(config->cudathreads[cudadev], NULL);
	}
	fprintf(stderr, "cuda terminated\n");
#endif

#ifdef USE_SPU
	unsigned spu;
	for (spu = 0; spu < config->nspus; spu++)
	{
		thread_join(config->sputhreads[spu], NULL);
	}
	fprintf(stderr, "SPUs terminated\n");
#endif


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
