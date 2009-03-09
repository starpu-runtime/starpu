#ifndef __DW_SPARSE_CG_H__
#define __DW_SPARSE_CG_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <common/blas.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <common/timing.h>

#include <datawizard/datawizard.h>

#define MAXITER	100000
#define EPSILON	0.0000001f

/* code parameters */
static uint32_t size = 33554432;
static unsigned usecpu = 0;
static unsigned blocks = 512;
static unsigned grids  = 8;

struct cg_problem {
	struct data_state_t *ds_matrixA;
	struct data_state_t *ds_vecx;
	struct data_state_t *ds_vecb;
	struct data_state_t *ds_vecr;
	struct data_state_t *ds_vecd;
	struct data_state_t *ds_vecq;

	sem_t *sem;
	
	float alpha;
	float beta;
	float delta_0;
	float delta_old;
	float delta_new;
	float epsilon;

	int i;
	unsigned size;
};

/* some useful functions */
static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-block") == 0) {
			char *argptr;
			blocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-grid") == 0) {
			char *argptr;
			grids = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu") == 0) {
			usecpu = 1;
		}
	}
}


static void print_results(float *result, unsigned size)
{
	printf("**** RESULTS **** \n");
	unsigned i;

	for (i = 0; i < MIN(size, 16); i++)
	{
		printf("%d -> %f\n", i, result[i]);
	}
}

static void create_data(float **_nzvalA, float **_vecb, float **_vecx, uint32_t *_nnz, uint32_t *_nrow, uint32_t **_colind, uint32_t **_rowptr)
{
	/* we need a sparse symetric (definite positive ?) matrix and a "dense" vector */
	
	/* example of 3-band matrix */
	float *nzval;
	uint32_t nnz;
	uint32_t *colind;
	uint32_t *rowptr;

	nnz = 3*size-2;

	nzval = malloc(nnz*sizeof(float));
	colind = malloc(nnz*sizeof(uint32_t));
	rowptr = malloc(size*sizeof(uint32_t));

	assert(nzval);
	assert(colind);
	assert(rowptr);


	/* fill the matrix */
	unsigned row;
	unsigned pos = 0;
	for (row = 0; row < size; row++)
	{
		rowptr[row] = pos;

		if (row > 0) {
			nzval[pos] = 1.0f;
			colind[pos] = row-1;
			pos++;
		}
		
		nzval[pos] = 5.0f;
		colind[pos] = row;
		pos++;

		if (row < size - 1) {
			nzval[pos] = 1.0f;
			colind[pos] = row+1;
			pos++;
		}
	}

	*_nnz = nnz;
	*_nrow = size;
	*_nzvalA = nzval;
	*_colind = colind;
	*_rowptr = rowptr;

	STARPU_ASSERT(pos == nnz);
	
	/* initiate the 2 vectors */
	float *invec, *outvec;
	invec = malloc(size*sizeof(float));
	assert(invec);

	outvec = malloc(size*sizeof(float));
	assert(outvec);

	/* fill those */
	unsigned ind;
	for (ind = 0; ind < size; ind++)
	{
		invec[ind] = 2.0f;
		outvec[ind] = 0.0f;
	}

	*_vecb = invec;
	*_vecx = outvec;
}

static job_t create_job(tag_t id)
{
	codelet *cl = malloc(sizeof(codelet));
		cl->cl_arg = NULL;
		cl->where = ANY;

	job_t j = job_create();
		j->cl = cl;

	tag_declare(id, j);

	return j;
}

void core_codelet_func_1(data_interface_t *descr, void *arg);

void core_codelet_func_2(data_interface_t *descr, void *arg);

void cublas_codelet_func_3(data_interface_t *descr, void *arg);
void core_codelet_func_3(data_interface_t *descr, void *arg);

void core_codelet_func_4(data_interface_t *descr, void *arg);

void core_codelet_func_5(data_interface_t *descr, void *arg);
void cublas_codelet_func_5(data_interface_t *descr, void *arg);

void cublas_codelet_func_6(data_interface_t *descr, void *arg);
void core_codelet_func_6(data_interface_t *descr, void *arg);

void cublas_codelet_func_7(data_interface_t *descr, void *arg);
void core_codelet_func_7(data_interface_t *descr, void *arg);

void cublas_codelet_func_8(data_interface_t *descr, void *arg);
void core_codelet_func_8(data_interface_t *descr, void *arg);

void cublas_codelet_func_9(data_interface_t *descr, void *arg);
void core_codelet_func_9(data_interface_t *descr, void *arg);

void iteration_cg(void *problem);

void conjugate_gradient(float *nzvalA, float *vecb, float *vecx, uint32_t nnz,
			unsigned nrow, uint32_t *colind, uint32_t *rowptr);

#endif // __DW_SPARSE_CG_H__
