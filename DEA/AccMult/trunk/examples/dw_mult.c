#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>

#include <datawizard/coherency.h>
#include <datawizard/hierarchy.h>
#include <datawizard/filters.h>

typedef struct { 
	data_state *subA;
	data_state *subB;
	data_state *subC;
} multdescr;

float *A;
float *B;
float *C;

data_state A_state;
data_state B_state;
data_state C_state;

/*
 * That program should compute C = A * B 
 * 
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)
 */

void callback_func(__attribute__ ((unused)) void *arg)
{
	/* the argument is a pointer to a counter of the remaining jobs */
	int *counter = arg;
	*counter = *counter - 1;

	if (*counter == 0) {
		/* we are done */	
		/* TODO unfilter */
		printf("done ...\n");
		exit(0);
	}

	return;
}

#define COMMON_CODE							\
	multdescr *descr = arg;						\
									\
	float *subA;							\
	float *subB;							\
	float *subC;							\
									\
	unsigned node = get_local_memory_node();			\
									\
	subA = (float *)fetch_data(descr->subA, node, 1, 0);		\
	subB = (float *)fetch_data(descr->subB, node, 1, 0);		\
	subC = (float *)fetch_data(descr->subC, node, 1, 1);		\
									\
	uint32_t nxC, nyC, nxA;						\
	nxC = get_local_nx(descr->subC) / sizeof(float);		\
	nyC = get_local_ny(descr->subC);				\
	nxA = get_local_nx(descr->subA) / sizeof(float);		\
									\
	uint32_t ldA, ldB, ldC;						\
	ldA = get_local_ld(descr->subA) / sizeof(float);		\
	ldB = get_local_ld(descr->subB) / sizeof(float);		\
	ldC = get_local_ld(descr->subC) / sizeof(float);		

#ifdef USE_CUBLAS
void cublas_mult(__attribute__ ((unused)) void *arg)
{
	COMMON_CODE

	cublasSgemm('n', 'n', nxC, nyC, nxA, 1.0f, subB, ldB, subA, ldA, 0.0f, subC, ldC);

	release_data(descr->subC, node, 1);
}
#endif

void core_mult(__attribute__ ((unused)) void *arg)
{
	COMMON_CODE

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nxC, nyC, nxA,
			 1.0f,  subA, ldA, subB, ldB, 0.0f, subC, ldC);

	release_data(descr->subC, node, 0);
}

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned x,y,z;
	unsigned nslices;

	int jobcounter;

	x = 4096;
	y = 4096;
	z = 2048;

	nslices = 8;
	jobcounter = 8;

	A = malloc(z*y*sizeof(float));
	B = malloc(x*z*sizeof(float));
	C = malloc(x*y*sizeof(float));

	/* fill the A and B matrices */
	/* TODO */

	/* start the runtime */
	init_machine();
	init_workers();

	monitor_new_data(&A_state, 0, (uintptr_t)A, z*sizeof(float), z*sizeof(float), y);
	monitor_new_data(&B_state, 0, (uintptr_t)B, x*sizeof(float), x*sizeof(float), z);
	monitor_new_data(&C_state, 0, (uintptr_t)C, x*sizeof(float), x*sizeof(float), y);

	filter f;
	f.filter_func = block_filter_func;
	f.filter_arg = nslices;
		
	partition_data(&B_state, &f);
	partition_data(&C_state, &f);

	/* this array will contain the list of jobs to be performed */
	multdescr jobarguments[nslices];

	/* partition the work into slices */
	unsigned task;
	for (task = 0; task < nslices; task++) 
	{
		/* A B[task] = C[task] */
		job_t j;
		codelet *cl = malloc(sizeof(codelet));

		jobarguments[task].subA = &A_state;
		jobarguments[task].subB = &B_state.children[task];
		jobarguments[task].subC = &C_state.children[task];

		cl->cl_arg = &jobarguments[task];
		cl->core_func = core_mult;
#ifdef USE_CUBLAS
		cl->cublas_func = cublas_mult;
#endif

		j = job_new();
		j->type = CODELET;
		j->where = ANY;
		j->cb = callback_func;
		j->argcb = &jobcounter;
		j->cl = cl;
		
		push_task(j);
	}

	sleep(100);

	return 0;
}
