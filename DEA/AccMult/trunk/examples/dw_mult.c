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
#include <common/timing.h>

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

tick_t start;
tick_t end;

unsigned x,y,z;

/*
 * That program should compute C = A * B 
 * 
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)
 */

void terminate(void)
{
	GET_TICK(end);

	unpartition_data(&C_state, 0);

	printf("Computation took %2.2f ms\n", TIMING_DELAY(start, end)/1000);

#ifdef CHECK_OUTPUT
	/* check results */
	/* compute C = C - AB */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, y, x, z,
			 -1.0f,  A, z, B, x, 1.0f, C, x);
		
	/* make sure C = 0 */
	float err;
	err = cblas_sasum(x*y, C, 1);	
	
	if (err < x*y*0.001) {
		printf("Results are OK\n");
	}
	else {
		printf("There were errors ... err = %f\n", err);
	}
#endif // CHECK_OUTPUT

	exit(0);
}

void callback_func(void *arg)
{
	/* the argument is a pointer to a counter of the remaining jobs */
	int *counter = arg;
	*counter -= 1;
	if (*counter == 0)
	{
		/* we are done */	
		printf("done ...\n");
		terminate();
	}

	return;
}

#define COMMON_CODE							\
	uint32_t nxC, nyC, nxA;						\
	uint32_t ldA, ldB, ldC;						\
									\
	float *subA;							\
	float *subB;							\
	float *subC;							\
									\
	multdescr *descr = arg;						\
									\
	subA = (float *)fetch_data(descr->subA, R);			\
	subB = (float *)fetch_data(descr->subB, R);			\
	subC = (float *)fetch_data(descr->subC, W);			\
									\
	nxC = get_local_nx(descr->subC);				\
	nyC = get_local_ny(descr->subC);				\
	nxA = get_local_nx(descr->subA);				\
									\
	ldA = get_local_ld(descr->subA);				\
	ldB = get_local_ld(descr->subB);				\
	ldC = get_local_ld(descr->subC);		

#ifdef USE_CUBLAS
void cublas_mult(void *arg)
{
	COMMON_CODE

	cublasSgemm('n', 'n', nxC, nyC, nxA, 1.0f, subB, ldB, subA, ldA, 0.0f, subC, ldC);

	release_data(descr->subC, 1);
}
#endif

void core_mult(void *arg)
{
	COMMON_CODE

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nyC, nxC, nxA,
			 1.0f,  subA, ldA, subB, ldB, 0.0f, subC, ldC);

	release_data(descr->subC, 0);
}

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned i,j;
	unsigned nslicesx, nslicesy;

	int jobcounter;


	timing_init();

	x = 4096;
	y = 4096;
	z = 4096;

	nslicesx = 4;
	nslicesy = 4;
	jobcounter = nslicesx * nslicesy;

	A = malloc(z*y*sizeof(float));
	B = malloc(x*z*sizeof(float));
	C = malloc(x*y*sizeof(float));

	/* fill the A and B matrices */
	srand(2008);
	for (i=0; i < z; i++) {
		for (j=0; j < y; j++) {
			A[i+j*z] = (float)(drand48());
		}
	}

	for (i=0; i < x; i++) {
		for (j=0; j < z; j++) {
			B[i+j*x] = (float)(drand48());
		}
	}

	for (i=0; i < x; i++) {
		for (j=0; j < y; j++) {
			C[i+j*x] = (float)(0);
		}
	}

	/* start the runtime */
	init_machine();
	init_workers();

	GET_TICK(start);
	monitor_new_data(&A_state, 0, (uintptr_t)A, z, z, y, sizeof(float));
	monitor_new_data(&B_state, 0, (uintptr_t)B, x, x, z, sizeof(float));
	monitor_new_data(&C_state, 0, (uintptr_t)C, x, x, y, sizeof(float));

	filter f;
	f.filter_func = block_filter_func;
	f.filter_arg = nslicesx;
		
	filter f2;
	f2.filter_func = vertical_block_filter_func;
	f2.filter_arg = nslicesy;
		
	partition_data(&B_state, &f);
	partition_data(&C_state, &f);


	partition_data(&A_state, &f2);
	map_filter(&C_state, &f2);

	/* this array will contain the list of jobs to be performed */
	multdescr jobarguments[nslicesx*nslicesy];

	/* partition the work into slices */
	unsigned taskx, tasky, task;
	job_t jb;

	task = 0;

	for (taskx = 0; taskx < nslicesx; taskx++) 
	{
		for (tasky = 0; tasky < nslicesy; tasky++)
		{
			/* A B[task] = C[task] */
			codelet *cl = malloc(sizeof(codelet));

			jobarguments[task].subA = &A_state.children[tasky];
			jobarguments[task].subB = &B_state.children[taskx];
			jobarguments[task].subC = &C_state.children[taskx].children[tasky];

			cl->cl_arg = &jobarguments[task];
			cl->core_func = core_mult;
#ifdef USE_CUBLAS
			cl->cublas_func = cublas_mult;
#endif

			jb = job_new();
			jb->type = CODELET;
			jb->where = ANY;
			jb->cb = callback_func;
			jb->argcb = &jobcounter;
			jb->cl = cl;
			
			push_task(jb);

			task++;
		}
	}

	sleep(100);

	return 0;
}
