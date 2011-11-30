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
	int *procs;
	int nprocs;
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

struct sched_ctx_hypervisor_reply reply1[NSAMPLES*2*2];
struct sched_ctx_hypervisor_reply reply2[NSAMPLES*2*2];

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

	if(p->ctx != 0)
	{
		pthread_mutex_lock(&mut);
		if(first){
		 	sched_ctx_hypervisor_ignore_ctx(p->ctx);
			starpu_delete_sched_ctx(p->ctx, p->the_other_ctx);
		}

		first = 0;
		pthread_mutex_unlock(&mut);
	}

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
//	sched_ctx_hypervisor_ignore_ctx(p1.ctx);
//	sched_ctx_hypervisor_ignore_ctx(p2.ctx);

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
	struct starpu_sched_ctx_hypervisor_criteria *criteria = sched_ctx_hypervisor_init(SIMPLE_POLICY);
	int nprocs1 = cpu1 + gpu + gpu1;
	int nprocs2 = cpu2 + gpu + gpu2;
	unsigned n_all_gpus = gpu + gpu1 + gpu2;


	int i;
	int k = 0;

	p1.procs = (int*)malloc(nprocs1*sizeof(int));

	for(i = 0; i < gpu; i++)
		p1.procs[k++] = i;

	for(i = gpu; i < gpu + gpu1; i++)
		p1.procs[k++] = i;


	for(i = n_all_gpus; i < n_all_gpus + cpu1; i++)
		p1.procs[k++] = i;


	p1.ctx = starpu_create_sched_ctx_with_criteria("heft", p1.procs, nprocs1, "sched_ctx1", criteria);
	p2.the_other_ctx = (int)p1.ctx;
	p1.nprocs = nprocs1;
	sched_ctx_hypervisor_handle_ctx(p1.ctx);
	
	sched_ctx_hypervisor_ioctl(p1.ctx,
				   HYPERVISOR_MAX_IDLE, p1.procs, p1.nprocs, 1000000.0,
				   HYPERVISOR_MAX_IDLE, p1.procs, gpu+gpu1, 100000000.0,
				   HYPERVISOR_MIN_WORKING, p1.procs, p1.nprocs, 200.0,
//				   HYPERVISOR_PRIORITY, p1.procs, p1.nprocs, 1,
				   HYPERVISOR_PRIORITY, p1.procs, gpu+gpu1, 2,
				   HYPERVISOR_MIN_PROCS, 0,
				   HYPERVISOR_MAX_PROCS, 11,
				   HYPERVISOR_GRANULARITY, 2,
//				   HYPERVISOR_FIXED_PROCS, p1.procs, gpu,
				   HYPERVISOR_MIN_TASKS, 10000,
				   HYPERVISOR_NEW_WORKERS_MAX_IDLE, 1000000.0,
				   NULL);

	k = 0;
	p2.procs = (int*)malloc(nprocs2*sizeof(int));

	for(i = 0; i < gpu; i++)
		p2.procs[k++] = i;

	for(i = gpu + gpu1; i < gpu + gpu1 + gpu2; i++)
		p2.procs[k++] = i;

	for(i = n_all_gpus  + cpu1; i < n_all_gpus + cpu1 + cpu2; i++)
		p2.procs[k++] = i;

	p2.ctx = starpu_create_sched_ctx_with_criteria("heft", p2.procs, nprocs2, "sched_ctx2", criteria);
	p1.the_other_ctx = (int)p2.ctx;
	p2.nprocs = nprocs2;
	sched_ctx_hypervisor_handle_ctx(p2.ctx);
	
	sched_ctx_hypervisor_ioctl(p2.ctx,
				   HYPERVISOR_MAX_IDLE, p2.procs, p2.nprocs, 100000.0,
				   HYPERVISOR_MAX_IDLE, p2.procs, gpu+gpu2, 10000000000.0,
				   HYPERVISOR_MIN_WORKING, p2.procs, p2.nprocs, 200.0,
//				   HYPERVISOR_PRIORITY, p2.procs, p2.nprocs, 1,
				   HYPERVISOR_PRIORITY, p2.procs, gpu+gpu2, 2,
				   HYPERVISOR_MIN_PROCS, 0,
				   HYPERVISOR_MAX_PROCS, 11,
				   HYPERVISOR_GRANULARITY, 4,
				   HYPERVISOR_FIXED_PROCS, p2.procs, gpu,
				   HYPERVISOR_MIN_TASKS, 10000,
				   HYPERVISOR_NEW_WORKERS_MAX_IDLE, 100000.0,
				   NULL);
}

void set_hypervisor_conf(int event, int task_tag)
{
	unsigned *id = pthread_getspecific(key);
	int reset_conf = 1;
	pthread_mutex_lock(&mut);
	reset_conf = first;
	pthread_mutex_unlock(&mut);


	if(reset_conf)
	{
		if(*id == 1)
		{			
			if(event == START_BENCH)
			{
				//sched_ctx_hypervisor_request(p2.ctx, p2.procs, p2.nprocs, task_tag);
				/* sched_ctx_hypervisor_ioctl(p2.ctx, */
				/* 			   HYPERVISOR_MAX_IDLE, p2.procs, p2.nprocs, 10000000.0, */
				/* 			   HYPERVISOR_TIME_TO_APPLY, task_tag, */
				/* 			   HYPERVISOR_GRANULARITY, 1, */
				/* 			   NULL); */
			}
		} else {

			if(event == START_BENCH)
			{
				int procs[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
				sched_ctx_hypervisor_request(p1.ctx, procs, 12, task_tag);
				/* sched_ctx_hypervisor_ioctl(p1.ctx, */
				/* 			   HYPERVISOR_MAX_IDLE, procs, 12, 1000.0, */
				/* 			   HYPERVISOR_MAX_IDLE, procs, 3, 1000000.0, */
				/* 			   HYPERVISOR_TIME_TO_APPLY, task_tag, */
				/* 			   HYPERVISOR_GRANULARITY, 4, */
				/* 			   NULL); */
			}
			else
			{
				/* int procs[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
				/* sched_ctx_hypervisor_ioctl(p1.ctx, */
				/* 			   HYPERVISOR_MAX_IDLE, procs, 12, 1000.0, */
				/* 			   HYPERVISOR_TIME_TO_APPLY, task_tag, */
				/* 			   HYPERVISOR_GRANULARITY, 4, */
				/* 			   NULL); */
			}

		}
	}
}

void end_contexts()
{
	free(p1.procs);
	free(p2.procs);
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

