#include "dw_factolu.h"
#include <core/tags.h>

tick_t start;
tick_t end;

#define TAG11(k)	( (1ULL<<60) | (unsigned long long)(k))
#define TAG12(k,i)	(((2ULL<<60) | (((unsigned long long)(k))<<32) | (unsigned long long)(i)))
#define TAG21(k,j)	(((3ULL<<60) | (((unsigned long long)(k))<<32) | (unsigned long long)(j)))
#define TAG22(k,i,j)	(((4ULL<<60) | ((unsigned long long)(k)<<32) | ((unsigned long long)(i)<<16) | (unsigned long long)(j)))


/* to compute MFlop/s */
static uint64_t flop_cublas = 0;
static uint64_t flop_atlas = 0;

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

/*
 *   U22 
 */

static inline void dw_common_core_codelet_update_u22(buffer_descr *buffers, int s, __attribute__((unused)) void *_args)
{
	float *left 	= (float *)buffers[0].ptr;
	float *right 	= (float *)buffers[1].ptr;
	float *center 	= (float *)buffers[2].ptr;

	unsigned dx = buffers[2].nx;
	unsigned dy = buffers[2].ny;
	unsigned dz = buffers[0].ny;

	unsigned ld12 = buffers[0].ld;
	unsigned ld21 = buffers[1].ld;
	unsigned ld22 = buffers[2].ld;

	switch (s) {
		case 0:
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				dy, dx, dz, -1.0f, left, ld21, right, ld12,
					     1.0f, center, ld22);
			flop_atlas += BLAS3_FLOP(dx, dy, dz);

			break;
#ifdef USE_CUBLAS
		case 1:
			cublasSgemm('n', 'n', dx, dy, dz, -1.0f, left, ld21,
					right, ld12, 1.0f, center, ld22);
			flop_cublas += BLAS3_FLOP(dx, dy, dz);

			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

static void dw_core_codelet_update_u22(buffer_descr *descr, void *_args)
{
	dw_common_core_codelet_update_u22(descr, 0, _args);
}

#ifdef USE_CUBLAS
static void dw_cublas_codelet_update_u22(buffer_descr *descr, void *_args)
{
	dw_common_core_codelet_update_u22(descr, 1, _args);
}
#endif// USE_CUBLAS

/*
 * U12
 */

static inline void dw_common_codelet_update_u12(buffer_descr *buffers, int s, __attribute__((unused)) void *_args) {
	float *sub11;
	float *sub12;

	sub11 = (float *)buffers[0].ptr;	
	sub12 = (float *)buffers[1].ptr;

	unsigned ld11 = buffers[0].ld;
	unsigned ld12 = buffers[1].ld;

	unsigned nx12 = buffers[1].nx;
	unsigned ny12 = buffers[1].ny;

	/* solve L11 U12 = A12 (find U12) */
	switch (s) {
		case 0:
			cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
					 nx12, ny12, 1.0f, sub11, ld11, sub12, ld12);
			break;
#ifdef USE_CUBLAS
		case 1:
			cublasStrsm('R', 'U', 'N', 'N', ny12, nx12,
					1.0f, sub11, ld11, sub12, ld12);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

static void dw_core_codelet_update_u12(buffer_descr *descr, void *_args)
{
	 dw_common_codelet_update_u12(descr, 0, _args);
}

#ifdef USE_CUBLAS
static void dw_cublas_codelet_update_u12(buffer_descr *descr, void *_args)
{
	 dw_common_codelet_update_u12(descr, 1, _args);
}
#endif // USE_CUBLAS

/* 
 * U21
 */

static inline void dw_common_codelet_update_u21(buffer_descr *buffers, int s, __attribute__((unused)) void *_args) {
	float *sub11;
	float *sub21;

	sub11 = (float *)buffers[0].ptr;
	sub21 = (float *)buffers[1].ptr;

	unsigned ld11 = buffers[0].ld;
	unsigned ld21 = buffers[1].ld;

	unsigned nx21 = buffers[1].nx;
	unsigned ny21 = buffers[1].ny;

	switch (s) {
		case 0:
			cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, 
				CblasUnit, nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#ifdef USE_CUBLAS
		case 1:
			cublasStrsm('L', 'L', 'N', 'U', ny21, nx21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

static void dw_core_codelet_update_u21(buffer_descr *descr, void *_args)
{
	 dw_common_codelet_update_u21(descr, 0, _args);
}

#ifdef USE_CUBLAS
static void dw_cublas_codelet_update_u21(buffer_descr *descr, void *_args)
{
	dw_common_codelet_update_u21(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void dw_common_codelet_update_u11(buffer_descr *descr, int s, __attribute__((unused)) void *_args) 
{
	float *sub11;

	sub11 = (float *)descr[0].ptr; 

	unsigned nx = descr[0].nx;
	unsigned ld = descr[0].ld;

	unsigned z;

	switch (s) {
		case 0:
			for (z = 0; z < nx; z++)
			{
				float pivot;
				pivot = sub11[z+z*ld];
				ASSERT(pivot != 0.0f);
		
				cblas_sscal(nx - z - 1, 1.0f/pivot, &sub11[(z+1)+z*ld], 1);
		
				cblas_sger(CblasRowMajor, nx - z - 1, nx - z - 1, -1.0f,
								&sub11[(z+1)+z*ld], 1,
								&sub11[z+(z+1)*ld], ld,
								&sub11[(z+1) + (z+1)*ld],ld);
		
			}
			break;
#ifdef USE_CUBLAS
		case 1:
			for (z = 0; z < nx; z++)
			{
				float pivot;
				/* ok that's dirty and ridiculous ... */
				cublasGetVector(1, sizeof(float), &sub11[z+z*ld], sizeof(float), &pivot, sizeof(float));

				ASSERT(pivot != 0.0f);
				
				cublasSscal(nx - z - 1, 1.0f/pivot, &sub11[(z+1)+z*ld], 1);
				
				cublasSger(nx - z - 1, nx - z - 1, -1.0f,
								&sub11[(z+1)+z*ld], 1,
								&sub11[z+(z+1)*ld], ld,
								&sub11[(z+1) + (z+1)*ld],ld);
			}
			break;
#endif
		default:
			ASSERT(0);
			break;
	}

}


static void dw_core_codelet_update_u11(buffer_descr *descr, void *_args)
{
	dw_common_codelet_update_u11(descr, 0, _args);
}

#ifdef USE_CUBLAS
static void dw_cublas_codelet_update_u11(buffer_descr *descr, void *_args)
{
	dw_common_codelet_update_u11(descr, 1, _args);
}
#endif// USE_CUBLAS

/*
 *	Construct the DAG
 */

static job_t create_job(tag_t id)
{
	codelet *cl = malloc(sizeof(codelet));
	cl->cl_arg = NULL;

	job_t j = job_new();
		j->type = CODELET;
		j->where = ANY;
		j->cb = NULL;
		j->argcb = NULL;;
		j->cl = cl;	

	tag_declare(id, &j);

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
#ifdef USE_CUBLAS
	job->cl->cublas_func = dw_cublas_codelet_update_u11;
#endif

	/* which sub-data is manipulated ? */
	job->nbuffers = 1;
		job->buffers[0].state = get_sub_data(dataA, 2, k, k);
		job->buffers[0].mode = RW;


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
#ifdef USE_CUBLAS
	job->cl->cublas_func = dw_cublas_codelet_update_u12;
#endif

	/* which sub-data is manipulated ? */
	job->nbuffers = 2;
		job->buffers[0].state = get_sub_data(dataA, 2, k, k); 
		job->buffers[0].mode = R;
		job->buffers[1].state = get_sub_data(dataA, 2, i, k); 
		job->buffers[1].mode = RW;

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
//	printf("task 21 k,j = %d,%d TAG = %llx\n", k,j, TAG21(k,j));
	
	job->cl->core_func = dw_core_codelet_update_u21;
#ifdef USE_CUBLAS
	job->cl->cublas_func = dw_cublas_codelet_update_u21;
#endif
	
	/* which sub-data is manipulated ? */
	job->nbuffers = 2;
		job->buffers[0].state = get_sub_data(dataA, 2, k, k); 
		job->buffers[0].mode = R;
		job->buffers[1].state = get_sub_data(dataA, 2, k, j); 
		job->buffers[1].mode = RW;

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
#ifdef USE_CUBLAS
	job->cl->cublas_func = dw_cublas_codelet_update_u22;
#endif

	/* which sub-data is manipulated ? */
	job->nbuffers = 3;
		job->buffers[0].state = get_sub_data(dataA, 2, i, k); 
		job->buffers[0].mode = R;
		job->buffers[1].state = get_sub_data(dataA, 2, k, j); 
		job->buffers[2].mode = R;
		job->buffers[2].state = get_sub_data(dataA, 2, i, j); 
		job->buffers[2].mode = RW;

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

	/* create a new codelet */
	codelet *cl = malloc(sizeof(codelet));
	cl_args *args = malloc(sizeof(cl_args));

	sem_t sem;

	sem_init(&sem, 0, 0U);

	job_t entry_job;

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

	unsigned n = get_local_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;
	fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

static void initialize_system(float **A, float **B, unsigned dim, unsigned pinned)
{
	init_machine();
	init_workers();

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
	monitor_new_data(&dataA, 0, (uintptr_t)matA, ld, size, size, sizeof(float));

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
