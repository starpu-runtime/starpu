#include "dw_cholesky.h"
#include "dw_cholesky_models.h"

/* A [ y ] [ x ] */
float *A[NMAXBLOCKS][NMAXBLOCKS];
data_handle A_state[NMAXBLOCKS][NMAXBLOCKS];

/*
 *	Some useful functions
 */

static job_t create_job(tag_t id)
{
	codelet *cl = malloc(sizeof(codelet));
		cl->where = ANY;

	job_t j = job_create();
		j->cl = cl;	
		j->cl_arg = NULL;

	tag_declare(id, j);

	return j;
}

static void terminal_callback(void *argcb)
{
	sem_t *sem = argcb;
	sem_post(sem);
}

/*
 *	Create the codelets
 */

static job_t create_task_11(unsigned k, unsigned nblocks, sem_t *sem)
{
//	printf("task 11 k = %d TAG = %llx\n", k, (TAG11(k)));

	job_t job = create_job(TAG11(k));

//	job->where = CORE;

	job->cl->core_func = chol_core_codelet_update_u11;
	job->cl->model = &chol_model_11;
#ifdef USE_CUDA
	job->cl->cublas_func = chol_cublas_codelet_update_u11;
#endif
#ifdef USE_GORDON
	job->cl->gordon_func = SPU_FUNC_POTRF;
#endif

	/* which sub-data is manipulated ? */
	job->nbuffers = 1;
		job->buffers[0].state = A_state[k][k];
		job->buffers[0].mode = RW;

	/* this is an important task */
	job->priority = MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0) {
		tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}

	/* the very last task must be notified */
	if (k == nblocks - 1) {
		job->cb = terminal_callback;
		job->argcb = sem;
	}

	return job;
}


static void create_task_21(unsigned k, unsigned j)
{
	job_t job = create_job(TAG21(k, j));
	
	job->cl->core_func = chol_core_codelet_update_u21;
	job->cl->model = &chol_model_21;
#ifdef USE_CUDA
	job->cl->cublas_func = chol_cublas_codelet_update_u21;
#endif
#ifdef USE_GORDON
	job->cl->gordon_func = SPU_FUNC_STRSM;
#endif

	/* which sub-data is manipulated ? */
	job->nbuffers = 2;
		job->buffers[0].state = A_state[k][k]; 
		job->buffers[0].mode = R;
		job->buffers[1].state = A_state[j][k]; 
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

static void create_task_22(unsigned k, unsigned i, unsigned j)
{
	job_t job = create_job(TAG22(k, i, j));
//	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j));

//	job->where = CORE;

	job->cl->core_func = chol_core_codelet_update_u22;
	job->cl->model = &chol_model_22;
#ifdef USE_CUDA
	job->cl->cublas_func = chol_cublas_codelet_update_u22;
#endif
#ifdef USE_GORDON
	job->cl->gordon_func = SPU_FUNC_SGEMM;
#endif

	/* which sub-data is manipulated ? */
	job->nbuffers = 3;
		job->buffers[0].state = A_state[i][k]; 
		job->buffers[0].mode = R;
		job->buffers[1].state = A_state[j][k]; 
		job->buffers[1].mode = R;
		job->buffers[2].state = A_state[j][i]; 
		job->buffers[2].mode = RW;

	if ( (i == k + 1) && (j == k +1) ) {
		job->priority = MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG21(k, i), TAG21(k, j));
	}
	else {
		tag_declare_deps(TAG22(k, i, j), 2, TAG21(k, i), TAG21(k, j));
	}
}



/*
 *	code to bootstrap the factorization 
 *	and construct the DAG
 */

static void dw_cholesky_no_stride(void)
{
	struct timeval start;
	struct timeval end;

	/* create a new codelet */
	sem_t sem;
	sem_init(&sem, 0, 0U);

	job_t entry_job = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		job_t job = create_task_11(k, nblocks, &sem);
		if (k == 0) {
			/* for now, we manually launch the first task .. XXX */
			entry_job = job;
		}
		
		for (j = k+1; j<nblocks; j++)
		{
			create_task_21(k, j);

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
					create_task_22(k, i, j);
			}
		}
	}

	/* schedule the codelet */
	gettimeofday(&start, NULL);
	submit_job(entry_job);

	/* stall the application until the end of computations */
	sem_wait(&sem);
	sem_destroy(&sem);
	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	double flop = (1.0f*size*size*size)/3.0f;
	fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

int main(int argc, char **argv)
{
	unsigned x, y;
	unsigned i, j;

	parse_args(argc, argv);
	assert(nblocks <= NMAXBLOCKS);

	fprintf(stderr, "BLOCK SIZE = %d\n", size / nblocks);

	init_machine();
	timing_init();

	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y) {
			A[y][x] = malloc(BLOCKSIZE*BLOCKSIZE*sizeof(float));
			assert(A[y][x]);
		}
	}


	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y) {
			posix_memalign((void **)&A[y][x], 128, BLOCKSIZE*BLOCKSIZE*sizeof(float));
			assert(A[y][x]);
		}
	}

	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1) ( + n In to make is stable ) 
	 * */
	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	if (x <= y) {
		for (i = 0; i < BLOCKSIZE; i++)
		for (j = 0; j < BLOCKSIZE; j++)
		{
			A[y][x][i*BLOCKSIZE + j] =
				(float)(1.0f/((float) (1.0+(x*BLOCKSIZE+i)+(y*BLOCKSIZE+j))));

			/* make it a little more numerically stable ... ;) */
			if ((x == y) && (i == j))
				A[y][x][i*BLOCKSIZE + j] += (float)(2*size);
		}
	}



	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y) {
			monitor_blas_data(&A_state[y][x], 0, (uintptr_t)A[y][x], 
				BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, sizeof(float));
		}
	}

	dw_cholesky_no_stride();

	terminate_machine();
	return 0;
}


