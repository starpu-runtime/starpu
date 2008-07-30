#include "dw_factolu.h"

#define TAG11(k)	( (1ULL<<60) | (unsigned long long)(k))
#define TAG12(k,i)	(((2ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i)))
#define TAG21(k,j)	(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j)))
#define TAG22(k,i,j)	(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j)))

/*
 *	Construct the DAG
 */

static job_t create_job(tag_t id)
{
	codelet *cl = malloc(sizeof(codelet));
	cl->cl_arg = NULL;

	job_t j = job_create();
		j->where = ANY;
		j->cl = cl;	

	tag_declare(id, j);

	return j;
}

static void terminal_callback(void *argcb)
{
	sem_t *sem = argcb;
	sem_post(sem);
}

static job_t create_task_11(data_state *dataA, unsigned k, unsigned nblocks, sem_t *sem)
{
//	printf("task 11 k = %d TAG = %llx\n", k, (TAG11(k)));

	job_t job = create_job(TAG11(k));

	job->cl->core_func = dw_core_codelet_update_u11;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job->cl->cublas_func = dw_cublas_codelet_update_u11;
#endif

	job->cost_model = task_11_cost;

	/* which sub-data is manipulated ? */
	job->nbuffers = 1;
		job->buffers[0].state = get_sub_data(dataA, 2, k, k);
		job->buffers[0].mode = RW;

	/* this is an important task */
	job->priority = MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0) {
		tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}

	/* the very last task must be notified */
	if (k == nblocks -1) {
		job->cb = terminal_callback;
		job->argcb = sem;
	}

	return job;
}

static void create_task_12(data_state *dataA, unsigned k, unsigned i)
{
	job_t job = create_job(TAG12(k, i));
//	printf("task 12 k,i = %d,%d TAG = %llx\n", k,i, TAG12(k,i));

	job->cl->core_func = dw_core_codelet_update_u12;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job->cl->cublas_func = dw_cublas_codelet_update_u12;
#endif

	job->cost_model = task_12_cost;

	/* which sub-data is manipulated ? */
	job->nbuffers = 2;
		job->buffers[0].state = get_sub_data(dataA, 2, k, k); 
		job->buffers[0].mode = R;
		job->buffers[1].state = get_sub_data(dataA, 2, i, k); 
		job->buffers[1].mode = RW;

	if (i == k+1) {
		job->priority = MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		tag_declare_deps(TAG12(k, i), 2, TAG11(k), TAG22(k-1, i, k));
	}
	else {
		tag_declare_deps(TAG12(k, i), 1, TAG11(k));
	}
}

static void create_task_21(data_state *dataA, unsigned k, unsigned j)
{
	job_t job = create_job(TAG21(k, j));
	
	job->cl->core_func = dw_core_codelet_update_u21;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job->cl->cublas_func = dw_cublas_codelet_update_u21;
#endif

	job->cost_model = task_21_cost;
	
	/* which sub-data is manipulated ? */
	job->nbuffers = 2;
		job->buffers[0].state = get_sub_data(dataA, 2, k, k); 
		job->buffers[0].mode = R;
		job->buffers[1].state = get_sub_data(dataA, 2, k, j); 
		job->buffers[1].mode = RW;

	if (j == k+1) {
		job->priority = MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		tag_declare_deps(TAG21(k, j), 2, TAG11(k), TAG22(k-1, k, j));
	}
	else {
		tag_declare_deps(TAG21(k, j), 1, TAG11(k));
	}
}

static void create_task_22(data_state *dataA, unsigned k, unsigned i, unsigned j)
{
	job_t job = create_job(TAG22(k, i, j));
//	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j));

	job->cl->core_func = dw_core_codelet_update_u22;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job->cl->cublas_func = dw_cublas_codelet_update_u22;
#endif

	job->cost_model = task_22_cost;

	/* which sub-data is manipulated ? */
	job->nbuffers = 3;
		job->buffers[0].state = get_sub_data(dataA, 2, i, k); 
		job->buffers[0].mode = R;
		job->buffers[1].state = get_sub_data(dataA, 2, k, j); 
		job->buffers[2].mode = R;
		job->buffers[2].state = get_sub_data(dataA, 2, i, j); 
		job->buffers[2].mode = RW;

	if ( (i == k + 1) && (j == k +1) ) {
		job->priority = MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG12(k, i), TAG21(k, j));
	}
	else {
		tag_declare_deps(TAG22(k, i, j), 2, TAG12(k, i), TAG21(k, j));
	}
}

/*
 *	code to bootstrap the factorization 
 */

static void dw_codelet_facto_v3(data_state *dataA, unsigned nblocks)
{
	tick_t start;
	tick_t end;

	/* create a new codelet */
	sem_t sem;
	sem_init(&sem, 0, 0U);

	job_t entry_job = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		job_t job = create_task_11(dataA, k, nblocks, &sem);
		if (k == 0) {
			/* for now, we manually launch the first task .. XXX */
			entry_job = job;
		}
		
		for (i = k+1; i<nblocks; i++)
		{
			create_task_12(dataA, k, i);
			create_task_21(dataA, k, i);
		}

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
				create_task_22(dataA, k, i, j);
			}
		}
	}

	/* schedule the codelet */
	GET_TICK(start);
	push_task(entry_job);

	/* stall the application until the end of computations */
	sem_wait(&sem);
	sem_destroy(&sem);
	GET_TICK(end);

	double timing = timing_delay(&start, &end);
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	unsigned n = get_blas_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;
	fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

static void initialize_system(float **A, float **B, unsigned dim, unsigned pinned)
{
	init_machine();

	timing_init();

	if (pinned)
	{
		malloc_pinned(A, B, dim);
	} 
	else {
		*A = malloc(dim*dim*sizeof(float));
		*B = malloc(dim*sizeof(float));
	}
}

void dw_factoLU_tag(float *matA, unsigned size, unsigned ld, unsigned nblocks)
{

#ifdef CHECK_RESULTS
	fprintf(stderr, "Checking results ...\n");
	float *Asaved;
	Asaved = malloc(ld*ld*sizeof(float));

	memcpy(Asaved, matA, ld*ld*sizeof(float));
#endif

	data_state dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	monitor_blas_data(&dataA, 0, (uintptr_t)matA, ld, size, size, sizeof(float));

	filter f;
		f.filter_func = block_filter_func;
		f.filter_arg = nblocks;

	filter f2;
		f2.filter_func = vertical_block_filter_func;
		f2.filter_arg = nblocks;

	map_filters(&dataA, 2, &f, &f2);

	dw_codelet_facto_v3(&dataA, nblocks);

#ifdef CHECK_RESULTS
	compare_A_LU(Asaved, matA, size, ld);
#endif

}
