#include "sched_ctx_utils.h"
#include <starpu.h>
#include "sched_ctx_hypervisor.h"
#define NSAMPLES 3

unsigned size1;
unsigned size2;
unsigned nblocks1;
unsigned nblocks2;
unsigned cpu1;
unsigned cpu2;
unsigned gpu;
unsigned gpu1;
unsigned gpu2;

typedef struct {
	unsigned id;
	unsigned ctx;
	int the_other_ctx;
	int *workers;
	int nworkers;
	void (*bench)(float*, unsigned, unsigned);
	unsigned size;
	unsigned nblocks;
	float *mat[NSAMPLES];
} params;

typedef struct {
	double flops;
	double avg_timing;
} retvals;

int first = 1;
pthread_mutex_t mut;
retvals rv[2];
params p1, p2;
int it = 0;
int it2 = 0;

pthread_key_t key;

void init()
{
	size1 = 4*1024;
	size2 = 4*1024;
	nblocks1 = 16;
	nblocks2 = 16;
	cpu1 = 0;
	cpu2 = 0;
	gpu = 0;
	gpu1 = 0;
	gpu2 = 0;

	rv[0].flops = 0.0;
	rv[1].flops = 0.0;
	rv[1].avg_timing = 0.0;
	rv[1].avg_timing = 0.0;

	p1.ctx = 0;
	p2.ctx = 0;

	p1.id = 0;
	p2.id = 1;
	pthread_key_create(&key, NULL);
}

void update_sched_ctx_timing_results(double flops, double avg_timing)
{
	unsigned *id = pthread_getspecific(key);
	rv[*id].flops += flops;
	rv[*id].avg_timing += avg_timing;
}

void* start_bench(void *val){
	params *p = (params*)val;
	int i;

	pthread_setspecific(key, &p->id);

	if(p->ctx != 0)
		starpu_set_sched_ctx(&p->ctx);

	for(i = 0; i < NSAMPLES; i++)
		p->bench(p->mat[i], p->size, p->nblocks);
	
	/* if(p->ctx != 0) */
	/* { */
	/* 	pthread_mutex_lock(&mut); */
	/* 	if(first){ */
	/* 		sched_ctx_hypervisor_unregiser_ctx(p->ctx); */
	/* 		starpu_delete_sched_ctx(p->ctx, p->the_other_ctx); */
	/* 	} */
		
	/* 	first = 0; */
	/* 	pthread_mutex_unlock(&mut); */
	/* } */
	sched_ctx_hypervisor_stop_resize(p->the_other_ctx);
	rv[p->id].flops /= NSAMPLES;
	rv[p->id].avg_timing /= NSAMPLES;
}

float* construct_matrix(unsigned size)
{
	float *mat;
	starpu_malloc((void **)&mat, (size_t)size*size*sizeof(float));

	unsigned i,j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			/* mat[j +i*size] = ((i == j)?1.0f*size:0.0f); */
		}
	}
	return mat;
}
void start_2benchs(void (*bench)(float*, unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	p1.nblocks = nblocks1;
	
	p2.bench = bench;
	p2.size = size2;
	p2.nblocks = nblocks2;

	int i;
	for(i = 0; i < NSAMPLES; i++)
	{
		p1.mat[i] = construct_matrix(p1.size);
		p2.mat[i] = construct_matrix(p2.size);
	}

	pthread_t tid[2];
	pthread_mutex_init(&mut, NULL);

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	pthread_create(&tid[0], NULL, (void*)start_bench, (void*)&p1);
	pthread_create(&tid[1], NULL, (void*)start_bench, (void*)&p2);
 
	pthread_join(tid[0], NULL);
	pthread_join(tid[1], NULL);

	gettimeofday(&end, NULL);

	pthread_mutex_destroy(&mut);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f %2.2f ", rv[0].flops, rv[1].flops);
	printf("%2.2f %2.2f %2.2f\n", rv[0].avg_timing, rv[1].avg_timing, timing);

}

void start_1stbench(void (*bench)(float*, unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	p1.nblocks = nblocks1;

	int i;
	for(i = 0; i < NSAMPLES; i++)
	{
		p1.mat[i] = construct_matrix(p1.size);
	}

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	start_bench((void*)&p1);

	gettimeofday(&end, NULL);

	pthread_mutex_destroy(&mut);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f ", rv[0].flops);
	printf("%2.2f %2.2f\n", rv[0].avg_timing, timing);
}

void start_2ndbench(void (*bench)(float*, unsigned, unsigned))
{
	p2.bench = bench;
	p2.size = size2;
	p2.nblocks = nblocks2;
	int i;
	for(i = 0; i < NSAMPLES; i++)
	{
		p2.mat[i] = construct_matrix(p2.size);
	}
	
	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	start_bench((void*)&p2);

	gettimeofday(&end, NULL);

	pthread_mutex_destroy(&mut);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f ", rv[1].flops);
	printf("%2.2f %2.2f\n", rv[1].avg_timing, timing);
}

void construct_contexts(void (*bench)(float*, unsigned, unsigned))
{
	struct starpu_performance_counters *perf_counters = sched_ctx_hypervisor_init(IDLE_POLICY);
	int nworkers1 = cpu1 + gpu + gpu1;
	int nworkers2 = cpu2 + gpu + gpu2;
	unsigned n_all_gpus = gpu + gpu1 + gpu2;


	int i;
	int k = 0;
	nworkers1 = 12;
	p1.workers = (int*)malloc(nworkers1*sizeof(int));

	/* for(i = 0; i < gpu; i++) */
	/* 	p1.workers[k++] = i; */

	/* for(i = gpu; i < gpu + gpu1; i++) */
	/* 	p1.workers[k++] = i; */


	/* for(i = n_all_gpus; i < n_all_gpus + cpu1; i++) */
	/* 	p1.workers[k++] = i; */


	for(i = 0; i < 12; i++)
		p1.workers[i] = i; 

	p1.ctx = starpu_create_sched_ctx_with_perf_counters("heft", p1.workers, nworkers1, "sched_ctx1", perf_counters);
	p2.the_other_ctx = (int)p1.ctx;
	p1.nworkers = nworkers1;
	sched_ctx_hypervisor_register_ctx(p1.ctx, 0.0);
	
	/* sched_ctx_hypervisor_ioctl(p1.ctx, */
	/* 			   HYPERVISOR_MAX_IDLE, p1.workers, p1.nworkers, 5000.0, */
	/* 			   HYPERVISOR_MAX_IDLE, p1.workers, gpu+gpu1, 100000.0, */
	/* 			   HYPERVISOR_EMPTY_CTX_MAX_IDLE, p1.workers, p1.nworkers, 500000.0, */
	/* 			   HYPERVISOR_GRANULARITY, 2, */
	/* 			   HYPERVISOR_MIN_TASKS, 1000, */
	/* 			   HYPERVISOR_NEW_WORKERS_MAX_IDLE, 100000.0, */
	/* 			   HYPERVISOR_MIN_WORKERS, 6, */
	/* 			   HYPERVISOR_MAX_WORKERS, 12, */
	/* 			   NULL); */

	sched_ctx_hypervisor_ioctl(p1.ctx,
				   HYPERVISOR_GRANULARITY, 2,
				   HYPERVISOR_MIN_TASKS, 1000,
				   HYPERVISOR_MIN_WORKERS, 6,
				   HYPERVISOR_MAX_WORKERS, 12,
				   NULL);

	k = 0;
	p2.workers = (int*)malloc(nworkers2*sizeof(int));

	/* for(i = 0; i < gpu; i++) */
	/* 	p2.workers[k++] = i; */

	/* for(i = gpu + gpu1; i < gpu + gpu1 + gpu2; i++) */
	/* 	p2.workers[k++] = i; */

	/* for(i = n_all_gpus  + cpu1; i < n_all_gpus + cpu1 + cpu2; i++) */
	/* 	p2.workers[k++] = i; */

	p2.ctx = starpu_create_sched_ctx_with_perf_counters("heft", p2.workers, 0, "sched_ctx2", perf_counters);
	p1.the_other_ctx = (int)p2.ctx;
	p2.nworkers = 0;
	sched_ctx_hypervisor_register_ctx(p2.ctx, 0.0);
	
	/* sched_ctx_hypervisor_ioctl(p2.ctx, */
	/* 			   HYPERVISOR_MAX_IDLE, p2.workers, p2.nworkers, 2000.0, */
	/* 			   HYPERVISOR_MAX_IDLE, p2.workers, gpu+gpu2, 5000.0, */
	/* 			   HYPERVISOR_EMPTY_CTX_MAX_IDLE, p1.workers, p1.nworkers, 500000.0, */
	/* 			   HYPERVISOR_GRANULARITY, 2, */
	/* 			   HYPERVISOR_MIN_TASKS, 500, */
	/* 			   HYPERVISOR_NEW_WORKERS_MAX_IDLE, 1000.0, */
	/* 			   HYPERVISOR_MIN_WORKERS, 4, */
	/* 			   HYPERVISOR_MAX_WORKERS, 8, */
	/* 			   NULL); */

	sched_ctx_hypervisor_ioctl(p2.ctx,
				   HYPERVISOR_GRANULARITY, 2,
				   HYPERVISOR_MIN_TASKS, 500,
				   HYPERVISOR_MIN_WORKERS, 0,
				   HYPERVISOR_MAX_WORKERS, 6,
				   NULL);

}

void set_hypervisor_conf(int event, int task_tag)
{
/* 	unsigned *id = pthread_getspecific(key); */
/* 	if(*id == 0) */
/* 	{ */
/* 		if(event == END_BENCH) */
/* 		{ */
/* 			if(it < 2) */
/* 			{ */
/* 				sched_ctx_hypervisor_ioctl(p2.ctx, */
/* 							   HYPERVISOR_MIN_WORKERS, 2, */
/* 							   HYPERVISOR_MAX_WORKERS, 4, */
/* 							   HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */

/* 				printf("%d: set max %d for tag %d\n", p2.ctx, 4, task_tag); */
/* 				sched_ctx_hypervisor_ioctl(p1.ctx, */
/* 							   HYPERVISOR_MIN_WORKERS, 6, */
/* 							   HYPERVISOR_MAX_WORKERS, 8, */
/* 							   HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p1.ctx, 8, task_tag); */
/* 				sched_ctx_hypervisor_resize(p1.ctx, task_tag); */
/* 			} */
/* 			if(it == 2) */
/* 			{ */
/* 				sched_ctx_hypervisor_ioctl(p2.ctx, */
/* 							   HYPERVISOR_MIN_WORKERS, 12, */
/* 							   HYPERVISOR_MAX_WORKERS, 12, */
/* 							   HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p2.ctx, 12, task_tag); */
/* 				sched_ctx_hypervisor_ioctl(p1.ctx, */
/* 							   HYPERVISOR_MIN_WORKERS, 0, */
/* 							   HYPERVISOR_MAX_WORKERS, 0, */
/* 							   HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p1.ctx, 0, task_tag); */
/* 				sched_ctx_hypervisor_resize(p1.ctx, task_tag); */
/* 			} */
/* 			it++; */
				
/* 		} */
/* 	} */
/* 	else */
/* 	{ */
/* 		if(event == END_BENCH) */
/* 		{ */
/* 			if(it2 < 3) */
/* 			{ */
/* 				sched_ctx_hypervisor_ioctl(p1.ctx, */
/* 							   HYPERVISOR_MIN_WORKERS, 6, */
/* 							   HYPERVISOR_MAX_WORKERS, 12, */
/* 							   HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p1.ctx, 12, task_tag); */
/* 				sched_ctx_hypervisor_ioctl(p2.ctx, */
/* 							   HYPERVISOR_MIN_WORKERS, 0, */
/* 							   HYPERVISOR_MAX_WORKERS, 0, */
/* 							   HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p2.ctx, 0, task_tag); */
/* 				sched_ctx_hypervisor_resize(p2.ctx, task_tag); */
/* 			} */
/* 			it2++; */
/* 		} */
/* 	} */

	/* if(*id == 1) */
	/* { */
	/* 	if(event == START_BENCH) */
	/* 	{ */
	/* 		int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 		sched_ctx_hypervisor_ioctl(p1.ctx, */
	/* 					   HYPERVISOR_MAX_IDLE, workers, 12, 800000.0, */
	/* 					   HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 					   NULL); */
	/* 	} */
	/* 	else */
	/* 	{ */
	/* 		if(it2 < 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sched_ctx_hypervisor_ioctl(p2.ctx, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 12, 500.0, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 3, 200.0, */
	/* 						   HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   NULL); */
	/* 		} */
	/* 		if(it2 == 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sched_ctx_hypervisor_ioctl(p2.ctx, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 12, 1000.0, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 3, 500.0, */
	/* 						   HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   HYPERVISOR_MAX_WORKERS, 12, */
	/* 						   NULL); */
	/* 		} */
	/* 		it2++; */
	/* 	} */
		
	/* } else { */
	/* 	if(event == START_BENCH) */
	/* 	{ */
	/* 		int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 		sched_ctx_hypervisor_ioctl(p1.ctx, */
	/* 					   HYPERVISOR_MAX_IDLE, workers, 12, 1500.0, */
	/* 					   HYPERVISOR_MAX_IDLE, workers, 3, 4000.0, */
	/* 					   HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 					   NULL); */
	/* 	} */
	/* 	if(event == END_BENCH) */
	/* 	{ */
	/* 		if(it < 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sched_ctx_hypervisor_ioctl(p1.ctx, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 12, 100.0, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 3, 5000.0, */
	/* 						   HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   NULL); */
	/* 		} */
	/* 		if(it == 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sched_ctx_hypervisor_ioctl(p1.ctx, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 12, 5000.0, */
	/* 						   HYPERVISOR_MAX_IDLE, workers, 3, 10000.0, */
	/* 						   HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   NULL); */
	/* 		} */
			
	/* 		it++; */
	/* 	} */

	/* } */
}

void end_contexts()
{
	free(p1.workers);
	free(p2.workers);
	sched_ctx_hypervisor_shutdown();
}

void parse_args_ctx(int argc, char **argv)
{
	init();
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size1") == 0) {
			char *argptr;
			size1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks1") == 0) {
			char *argptr;
			nblocks1 = strtol(argv[++i], &argptr, 10);
		}
		
		if (strcmp(argv[i], "-size2") == 0) {
			char *argptr;
			size2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks2") == 0) {
			char *argptr;
			nblocks2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu1") == 0) {
			char *argptr;
			cpu1 = strtol(argv[++i], &argptr, 10);
		}    

		if (strcmp(argv[i], "-cpu2") == 0) {
			char *argptr;
			cpu2 = strtol(argv[++i], &argptr, 10);
		}    

		if (strcmp(argv[i], "-gpu") == 0) {
			char *argptr;
			gpu = strtol(argv[++i], &argptr, 10);
		}    

		if (strcmp(argv[i], "-gpu1") == 0) {
			char *argptr;
			gpu1 = strtol(argv[++i], &argptr, 10);
		}    

		if (strcmp(argv[i], "-gpu2") == 0) {
			char *argptr;
			gpu2 = strtol(argv[++i], &argptr, 10);
		}    
	}
}

