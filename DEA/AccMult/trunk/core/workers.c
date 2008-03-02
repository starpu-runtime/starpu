#include <common/timing.h>
#include "workers.h"

/* number of actual CPU cores */

#ifdef USE_CPUS
unsigned ncores;
thread_t corethreads[NMAXCORES];
core_worker_arg coreargs[NMAXCORES]; 
#endif

#ifdef USE_CUDA
thread_t cudathreads[MAXCUDADEVS];
int cudacounters[MAXCUDADEVS];
cuda_worker_arg cudaargs[MAXCUDADEVS];
extern int ncudagpus;
#endif

#ifdef USE_CUBLAS
thread_t cublasthreads[MAXCUBLASDEVS];
int cublascounters[MAXCUBLASDEVS];
cublas_worker_arg cublasargs[MAXCUBLASDEVS];
extern int ncublasgpus;
#endif

#ifdef USE_CPUS
int corecounters[NMAXCORES];
#endif

static int current_bindid = 0;

void init_machine(void)
{
	srand(2008);

#ifdef USE_CPUS
	ncores = MIN(sysconf(_SC_NPROCESSORS_ONLN), NMAXCORES);
#endif

#ifdef USE_CUDA
	init_cuda();
#endif
	timing_init();
}

void init_workers(void) 
{
	/* initialize the queue containing the jobs */
	init_work_queue();

	/* launch one thread per CPU */
#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		corecounters[core] = 0;

		coreargs[core].bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));
		
		coreargs[core].coreid = core;
		coreargs[core].ready_flag = 0;

		thread_create(&corethreads[core], NULL, core_worker, &coreargs[core]);
		/* wait until the thread is actually launched ... */
		while (coreargs[core].ready_flag == 0) {}
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < ncudagpus; cudadev++)
	{
		cudaargs[cudadev].deviceid = cudadev;
		cudaargs[cudadev].ready_flag = 0;

		cudaargs[cudadev].bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		cudacounters[cudadev] = 0;

		thread_create(&cudathreads[cudadev], NULL, cuda_worker, (void*)&cudaargs[cudadev]);

		/* wait until the thread is actually launched ... */
		while (cudaargs[cudadev].ready_flag == 0) {}
	}
#endif


#ifdef USE_CUBLAS
	/* initialize CUBLAS with the proper number of threads */
	int cublasdev;
	for (cublasdev = 0; cublasdev < ncublasgpus; cublasdev++)
	{
		cublasargs[cublasdev].deviceid = cublasdev;
		cublasargs[cublasdev].ready_flag = 0;

		cublasargs[cublasdev].bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		cublascounters[cublasdev] = 0;

		thread_create(&cublasthreads[cublasdev], NULL, cublas_worker, (void*)&cublasargs[cublasdev]);

		/* wait until the thread is actually launched ... */
		while (cublasargs[cublasdev].ready_flag == 0) {}
	}
#endif

}

void terminate_workers(void)
{
	printf("terminate workers \n");
#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		thread_join(corethreads[core], NULL);
	}
	printf("core terminated ... \n");
#endif



#ifdef USE_CUDA
	int cudadev;
	for (cudadev = 0; cudadev < ncudagpus; cudadev++)
	{
		thread_join(cudathreads[cudadev], NULL);
	}
	printf("cuda terminated\n");
#endif

#ifdef USE_CUBLAS
	int cublasdev;
	for (cublasdev = 0; cublasdev < ncublasgpus; cublasdev++)
	{
		thread_join(cublasthreads[cublasdev], NULL);
	}
	printf("cublas terminated\n");
#endif

}

void kill_all_workers(void)
{
        /* terminate all threads */
        unsigned nworkers = 0;

#ifdef USE_CPUS
        nworkers += ncores;
#endif
#ifdef USE_CUDA
        nworkers += ncudagpus;
#endif
#ifdef USE_CUBLAS
        nworkers += ncublasgpus;
#endif

        unsigned worker;
        for (worker = 0; worker < nworkers ; worker++) {
                job_t j = job_new();
                j->type = ABORT;
                j->where = ANY;
                push_task(j);
        }

        if (nworkers == 0) {
                fprintf(stderr, "Warning there is no worker ... \n");
        }

}


int count_tasks(void)
{
	int total = 0;
	unsigned i __attribute__ ((unused));

#ifdef USE_CPUS
	for (i = 0; i < ncores ; i++)
	{
		total += corecounters[i];
	}
#endif

#ifdef USE_CUDA
	for (i = 0; i < ncudagpus ; i++)
	{
		total += cudacounters[i];
	}
#endif

#ifdef USE_CUBLAS
	for (i = 0; i < ncublasgpus ; i++)
	{
		total += cublascounters[i];
	}
#endif

	return total;
}

void display_general_stats()
{
	unsigned i __attribute__ ((unused));
	int total __attribute__ ((unused));
	
	total = count_tasks();

#ifdef USE_CPUS
	printf("CORES :\n");
	for (i = 0; i < ncores ; i++)
	{
		printf("\tcore %d\t %d tasks\t%f %%\n", i, corecounters[i],
							(100.0*corecounters[i])/total);
	}
#endif

#ifdef USE_CUDA
	printf("CUDA :\n");
	for (i = 0; i < ncudagpus ; i++)
	{
		printf("\tdev %d\t %d tasks\t%f %%\n", i, cudacounters[i],
							(100.0*cudacounters[i])/total);
	}
#endif

#ifdef USE_CUBLAS
	printf("CUBLAS :\n");
	for (i = 0; i < ncublasgpus ; i++)
	{
		printf("\tblas %d\t %d tasks\t%f %%\n", i, cublascounters[i],
							(100.0*cublascounters[i])/total);
	}
#endif
}

void display_stats(job_descr *jd)
{

#ifdef COMPARE_SEQ
	float refchrono	=  ((float)(TIMING_DELAY(jd->job_refstart, jd->job_refstop)));
	printf("Ref time : %f ms\n", refchrono/1000);
	printf("Speedup\t=\t%f\n", refchrono/chrono); 
#endif

	float chrono 	=  (float)(TIMING_DELAY(jd->job_submission, jd->job_finished));
	printf("Computation time : %f ms\n", chrono/1000);


}
