#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>

#include "dw_factolu.h"

tick_t start;
tick_t end;

/*
 *   U22 
 */

static inline void dw_common_core_codelet_update_u22(int s, void *_args)
{
	cl_args *args = _args;

	unsigned k = args->k;
	unsigned i = args->i;
	unsigned j = args->j;

	data_state *dataA = args->dataA;

	data_state *data12 = get_sub_data(dataA, 2, i, k);
	data_state *data21 = get_sub_data(dataA, 2, k, j);
	data_state *data22 = get_sub_data(dataA, 2, i, j);

	float *left 	= (float *)fetch_data(data21, R); 
	float *right 	= (float *)fetch_data(data12, R); 
	float *center 	= (float *)fetch_data(data22, RW);

	unsigned dx = get_local_nx(data22);
	unsigned dy = get_local_ny(data22);
	unsigned dz = get_local_ny(data12);

	unsigned ld12 = get_local_ld(data12);
	unsigned ld21 = get_local_ld(data21);
	unsigned ld22 = get_local_ld(data22);

	switch (s) {
		case 0:
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				dy, dx, dz, -1.0f, left, ld21, right, ld12,
					     1.0f, center, ld22);
			release_data(data22, 0);
			break;
#ifdef USE_CUBLAS
		case 1:
			cublasSgemm('n', 'n', dx, dy, dz, -1.0f, left, ld21,
					right, ld12, 1.0f, center, ld22);
			release_data(data22, 1<<0);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}

	/* data that were only read */
	release_data(data12, 0);
	release_data(data21, 0);
}

void dw_core_codelet_update_u22(void *_args)
{
	dw_common_core_codelet_update_u22(0, _args);
}

#ifdef USE_CUBLAS
void dw_cublas_codelet_update_u22(void *_args)
{
	dw_common_core_codelet_update_u22(1, _args);
}
#endif// USE_CUBLAS

/*
 * U12
 */

static inline void dw_common_codelet_update_u12(int s, void *_args) {
	float *sub11;
	float *sub12;

	cl_args *args = _args;

	unsigned i = args->i;
	unsigned k = args->k;

	data_state *dataA = args->dataA;

	data_state *data11 = get_sub_data(dataA, 2, i, i);
	data_state *data12 = get_sub_data(dataA, 2, k, i);

	sub11 = (float *)fetch_data(data11, R);	
	sub12 = (float *)fetch_data(data12, RW);

	unsigned ld11 = get_local_ld(data11);
	unsigned ld12 = get_local_ld(data12);

	unsigned nx12 = get_local_nx(data12);
	unsigned ny12 = get_local_ny(data12);

	/* solve L11 U12 = A12 (find U12) */
	switch (s) {
		case 0:
			cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
					 nx12, ny12, 1.0f, sub11, ld11, sub12, ld12);
			release_data(data12, 0);
			break;
#ifdef USE_CUBLAS
		case 1:
			cublasStrsm('R', 'U', 'N', 'N', ny12, nx12,
					1.0f, sub11, ld11, sub12, ld12);
			release_data(data12, 1<<0);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}

	release_data(data11, 0);
}

void dw_core_codelet_update_u12(void *_args)
{
	 dw_common_codelet_update_u12(0, _args);
}

#ifdef USE_CUBLAS
void dw_cublas_codelet_update_u12(void *_args)
{
	 dw_common_codelet_update_u12(1, _args);
}
#endif // USE_CUBLAS

/* 
 * U21
 */

static inline void dw_common_codelet_update_u21(int s, void *_args) {
	float *sub11;
	float *sub21;

	cl_args *args = _args;

	unsigned i = args->i;
	unsigned k = args->k;

	data_state *dataA = args->dataA;

	data_state *data11 = get_sub_data(dataA, 2, i, i);
	data_state *data21 = get_sub_data(dataA, 2, i, k);
	
	sub11 = (float *)fetch_data(data11, R);
	sub21 = (float *)fetch_data(data21, RW);

	unsigned ld11 = get_local_ld(data11);
	unsigned ld21 = get_local_ld(data21);

	unsigned nx21 = get_local_nx(data21);
	unsigned ny21 = get_local_ny(data21);

	switch (s) {
		case 0:
			cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, 
				CblasUnit, nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			release_data(data21, 0);
			break;
#ifdef USE_CUBLAS
		case 1:
			cublasStrsm('L', 'L', 'N', 'U', ny21, nx21, 1.0f, sub11, ld11, sub21, ld21);
			release_data(data21, 1<<0);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}

	release_data(data11, 0);
}

void dw_core_codelet_update_u21(void *_args)
{
	 dw_common_codelet_update_u21(0, _args);
}

#ifdef USE_CUBLAS
void dw_cublas_codelet_update_u21(void *_args)
{
	dw_common_codelet_update_u21(1, _args);
}
#endif 

/*
 *	U11
 */

void dw_core_codelet_update_u11(void *_args)
{
	float *sub11;
	cl_args *args = _args;

	unsigned i = args->i;
	data_state *dataA = args->dataA;

	data_state *subdata11 = get_sub_data(dataA, 2, i, i);
	sub11 = (float *)fetch_data(subdata11, RW); 

	unsigned nx = get_local_nx(subdata11);
	unsigned ld = get_local_ld(subdata11);

	unsigned x, y, z;

	for (z = 0; z < nx; z++)
	{
		float pivot;
		pivot = sub11[z+z*ld];
		ASSERT(pivot != 0.0f);
		for (x = z+1; x < nx ; x++)
		{
			sub11[x+z*ld] = sub11[x+z*ld] / pivot;
		}

		for (y = z+1; y < nx; y++)
		{
			float tmp = sub11[z+y*ld];
			for (x = z+1; x < nx ; x++)
			{
				sub11[x+y*ld] -= sub11[x+z*ld]*tmp;
			}
		}
	}

	release_data(subdata11, 0);
}


/*
 *	Callbacks 
 */

void dw_callback_codelet_update_u22(void *argcb)
{
	cl_args *args = argcb;	

	if (ATOMIC_ADD(args->remaining, (-1)) == 0)
	{
		/* all worker already used the counter */
		free(args->remaining);

		/* we now reduce the LU22 part (recursion appears there) */
		codelet *cl = malloc(sizeof(codelet));
		cl_args *u11arg = malloc(sizeof(cl_args));
	
		cl->cl_arg = u11arg;
		cl->core_func = dw_core_codelet_update_u11;
	
		job_t j = job_new();
			j->type = CODELET;
			j->where = CORE;
			j->cb = dw_callback_codelet_update_u11;
			j->argcb = u11arg;
			j->cl = cl;
	
		u11arg->dataA = args->dataA;
		u11arg->i = args->k + 1;
		u11arg->nblocks = args->nblocks;
		u11arg->sem = args->sem;

		/* schedule the codelet */
		push_task(j);

	}

	free(args);
}

void dw_callback_codelet_update_u12_21(void *argcb)
{
	cl_args *args = argcb;	

	if (ATOMIC_ADD(args->remaining, -1) == 0)
	{
		/* now launch the update of LU22 */
		unsigned i = args->i;
		unsigned nblocks = args->nblocks;

		/* the number of jobs to be done */
		unsigned *remaining = malloc(sizeof(unsigned));
		*remaining = (nblocks - 1 - i)*(nblocks - 1 - i);

		unsigned slicey, slicex;
		for (slicey = i+1; slicey < nblocks; slicey++)
		{
			for (slicex = i+1; slicex < nblocks; slicex++)
			{
				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));
				codelet *cl22 = malloc(sizeof(codelet));

				cl22->cl_arg = u22a;
				cl22->core_func = dw_core_codelet_update_u22;
#ifdef USE_CUBLAS
				cl22->cublas_func = dw_cublas_codelet_update_u22;
#endif

				job_t j22 = job_new();
				j22->type = CODELET;
				j22->where = ANY;
				j22->cb = dw_callback_codelet_update_u22;
				j22->argcb = u22a;
				j22->cl = cl22;

				u22a->k = i;
				u22a->i = slicex;
				u22a->j = slicey;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;
				u22a->remaining = remaining;
				u22a->sem = args->sem;
				
				/* schedule that codelet */
				push_task(j22);
			}
		}
	}
}



void dw_callback_codelet_update_u11(void *argcb)
{
	/* in case there remains work, go on */
	cl_args *args = argcb;

	if (args->i == args->nblocks - 1) 
	{
		/* we are done : wake the application up  */
		sem_post(args->sem);
		return;
	}
	else 
	{
		/* put new tasks */
		unsigned nslices;
		nslices = args->nblocks - 1 - args->i;

		unsigned *remaining = malloc(sizeof(unsigned));
		*remaining = 2*nslices; 

		unsigned slice;
		for (slice = args->i + 1; slice < args->nblocks; slice++)
		{

			/* update slice from u12 */
			cl_args *u12a = malloc(sizeof(cl_args));
			codelet *cl12 = malloc(sizeof(codelet));

			/* update slice from u21 */
			cl_args *u21a = malloc(sizeof(cl_args));
			codelet *cl21 = malloc(sizeof(codelet));

			cl12->cl_arg = u12a;
			cl21->cl_arg = u21a;
			cl12->core_func = dw_core_codelet_update_u12;
			cl21->core_func = dw_core_codelet_update_u21;
#ifdef USE_CUBLAS
			cl12->cublas_func = dw_cublas_codelet_update_u12;
			cl21->cublas_func = dw_cublas_codelet_update_u21;
#endif

			job_t j12 = job_new();
				j12->type = CODELET;
				j12->where = ANY;
				j12->cb = dw_callback_codelet_update_u12_21;
				j12->argcb = u12a;
				j12->cl = cl12;

			job_t j21 = job_new();
				j21->type = CODELET;
				j21->where = ANY;
				j21->cb = dw_callback_codelet_update_u12_21;
				j21->argcb = u21a;
				j21->cl = cl21;
			

			u12a->i = args->i;
			u12a->k = slice;
			u12a->nblocks = args->nblocks;
			u12a->dataA = args->dataA;
			u12a->remaining = remaining;
			u12a->sem = args->sem;
			
			u21a->i = args->i;
			u21a->k = slice;
			u21a->nblocks = args->nblocks;
			u21a->dataA = args->dataA;
			u21a->remaining = remaining;
			u21a->sem = args->sem;

			push_task(j12);
			push_task(j21);
		}
	}
}

/*
 *	code to bootstrap the factorization 
 */

void dw_codelet_facto(data_state *dataA, unsigned nblocks)
{

	/* create a new codelet */
	codelet *cl = malloc(sizeof(codelet));
	cl_args *args = malloc(sizeof(cl_args));

	sem_t sem;

	sem_init(&sem, 0, 0U);

	args->sem = &sem;
	args->i = 0;
	args->nblocks = nblocks;
	args->dataA = dataA;

	cl->cl_arg = args;
	cl->core_func = dw_core_codelet_update_u11;

	GET_TICK(start);

	/* inject a new task with this codelet into the system */ 
	job_t j = job_new();
		j->type = CODELET;
		j->where = CORE;
		j->cb = dw_callback_codelet_update_u11;
		j->argcb = args;
		j->cl = cl;

	/* schedule the codelet */
	push_task(j);

	/* stall the application until the end of computations */
	sem_wait(&sem);
	sem_destroy(&sem);
	GET_TICK(end);
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", TIMING_DELAY(start, end)/1000);
}

void dw_factoLU(float *matA, unsigned size, unsigned ld, unsigned nblocks)
{
	init_machine();
	init_workers();

	timing_init();

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

	dw_codelet_facto(&dataA, nblocks);

#ifdef CHECK_RESULTS
	compare_A_LU(Asaved, matA, size, ld);
#endif

}
