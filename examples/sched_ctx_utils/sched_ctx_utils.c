#include "sched_ctx_utils.h"
#include <starpu.h>

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
	void (*bench)(unsigned, unsigned);
	unsigned size;
	unsigned nblocks;
} params;

typedef struct {
	double flops;
	double avg_timing;
} retvals;

#define NSAMPLES 3
int first = 1;
pthread_mutex_t mut;
retvals rv[2];
params p1, p2;

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
		p->bench(p->size, p->nblocks);

	if(p->ctx != 0)
	{
		pthread_mutex_lock(&mut);
		if(first){
			starpu_delete_sched_ctx(p->ctx, p->the_other_ctx);
		}
		
		first = 0;
		pthread_mutex_unlock(&mut);
	}

	rv[p->id].flops /= NSAMPLES;
	rv[p->id].avg_timing /= NSAMPLES;
	
}

void start_2benchs(void (*bench)(unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	p1.nblocks = nblocks1;
	
	p2.bench = bench;
	p2.size = size2;
	p2.nblocks = nblocks2;
	
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

void start_1stbench(void (*bench)(unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	p1.nblocks = nblocks1;
	
	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	start_bench((void*)&p1);

	gettimeofday(&end, NULL);

	pthread_mutex_destroy(&mut);
  
	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f %2.2f ", rv[0].flops);
	printf("%2.2f %2.2f %2.2f\n", rv[0].avg_timing, timing);
}

void start_2ndbench(void (*bench)(unsigned, unsigned))
{
	p2.bench = bench;
	p2.size = size2;
	p2.nblocks = nblocks2;
	
	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	start_bench((void*)&p2);

	gettimeofday(&end, NULL);

	pthread_mutex_destroy(&mut);
  
	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f %2.2f ", rv[1].flops);
	printf("%2.2f %2.2f %2.2f\n", rv[1].avg_timing, timing);
}

void construct_contexts(void (*bench)(unsigned, unsigned))
{
	int nprocs1 = cpu1 + gpu + gpu1;
	int nprocs2 = cpu2 + gpu + gpu2;
	unsigned n_all_gpus = gpu + gpu1 + gpu2;


	int procs[nprocs1];
	int i;
	int k = 0;

	for(i = 0; i < gpu; i++)
		procs[k++] = i;

	for(i = gpu; i < gpu + gpu1; i++)
		procs[k++] = i;


	for(i = n_all_gpus; i < n_all_gpus + cpu1; i++)
		procs[k++] = i;


	p1.ctx = starpu_create_sched_ctx("heft", procs, nprocs1, "sched_ctx1");
	p2.the_other_ctx = (int)p1.ctx;
	p1.procs = procs;
	p1.nprocs = nprocs1;
	int procs2[nprocs2];

	k = 0;

	for(i = 0; i < gpu; i++)
		procs2[k++] = i;

	for(i = gpu + gpu1; i < gpu + gpu1 + gpu2; i++)
		procs2[k++] = i;

	for(i = n_all_gpus  + cpu1; i < n_all_gpus + cpu1 + cpu2; i++)
		procs2[k++] = i;

	p2.ctx = starpu_create_sched_ctx("heft", procs2, nprocs2, "sched_ctx2");
	p1.the_other_ctx = (int)p2.ctx;
	p2.procs = procs2;
	p2.nprocs = nprocs2;
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

