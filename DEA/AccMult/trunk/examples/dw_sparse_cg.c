/*
 * Conjugate gradients for Sparse matrices
 */

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>
#include <common/timing.h>

#include <datawizard/coherency.h>
#include <datawizard/hierarchy.h>
#include <datawizard/interfaces/blas_interface.h>
#include <datawizard/interfaces/blas_filters.h>
#include <datawizard/interfaces/csr_interface.h>


/* First a Matrix-Vector product (SpMV) */

sem_t sem;
uint32_t size = 1024;

data_state sparse_matrix;
data_state vector_in, vector_out;

float *sparse_matrix_nzval;
uint32_t *sparse_matrix_colind;
uint32_t *sparse_matrix_rowptr;

float *vector_in_ptr;
float *vector_out_ptr;

void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}
	}
}

void core_spmv(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	printf("CORE codelet\n");

	/* for testing purprose ... */
	float *nzval;
	uint32_t nnz;

	nzval = (float *)descr[0].csr.nzval;
	nnz = descr[0].csr.nnz;

	cblas_sscal(nnz, 0.5f, nzval, 1);
}

void cublas_spmv(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	printf("CUBLAS codelet\n");
	
	/* for testing purprose ... */
	float *nzval;
	uint32_t nnz;

	nzval = (float *)descr[0].csr.nzval;
	nnz = descr[0].csr.nnz;

	cublasSscal(nnz, 0.5f, nzval, 1);
}

void create_data(void)
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

	ASSERT(pos == nnz);
	
	monitor_csr_data(&sparse_matrix, 0, nnz, size, (uintptr_t)nzval, colind, rowptr, 0, sizeof(float));

	sparse_matrix_nzval = nzval;
	sparse_matrix_colind = colind;
	sparse_matrix_rowptr = rowptr;

	printf("BEFORE sparse_matrix_nzval[0] = %2.2f\n", sparse_matrix_nzval[0]);

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

	monitor_blas_data(&vector_in, 0, (uintptr_t)invec, size, size, 1, sizeof(float));
	monitor_blas_data(&vector_out, 0, (uintptr_t)outvec, size, size, 1, sizeof(float));

	vector_in_ptr = invec;
	vector_out_ptr = outvec;
}

void init_problem_callback(void *arg __attribute__((unused)))
{
	sem_post(&sem);

	printf("AFTER sparse_matrix_nzval[0] = %2.2f\n", sparse_matrix_nzval[0]);
}

void call_spmv_codelet(void)
{
	job_t job;
	codelet *cl = malloc(sizeof(codelet));

	cl->cl_arg = NULL;
	cl->core_func =  core_spmv;
	cl->cublas_func = cublas_spmv;

	job = job_create();
	job->where = CUBLAS;
	job->cb = init_problem_callback;
	job->argcb = NULL;
	job->cl = cl;

	job->nbuffers = 3;
	job->buffers[0].state = &sparse_matrix;
	job->buffers[0].mode  = R;
	job->buffers[1].state = &vector_in;
	job->buffers[1].mode = R;
	job->buffers[2].state = &vector_out;
	job->buffers[2].mode = W;

	push_task(job);
}

void init_problem(void)
{
	/* create the sparse input matrix */
	create_data();

	/* create a new codelet that will perform a SpMV on it */
	call_spmv_codelet();
}

int main(__attribute__ ((unused)) int argc,
	__attribute__ ((unused)) char **argv)
{
	parse_args(argc, argv);

	/* start the runtime */
	init_machine();
	init_workers();

	sem_init(&sem, 0, 0U);

	init_problem();

	sem_wait(&sem);
	sem_destroy(&sem);

	return 0;
}
