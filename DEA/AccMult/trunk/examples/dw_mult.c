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

#include <common/fxt.h>

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


/* to compute MFlop/s */
uint64_t flop_cublas = 0;
uint64_t flop_atlas = 0;

/* to compute MB/s (load/store) */
uint64_t ls_cublas = 0;
uint64_t ls_atlas = 0;

#define BLAS3_FLOP(n1,n2,n3)	\
	(2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

#define BLAS3_LS(n1,n2,n3)    \
	(2*(n1)*(n3) + (n1)*(n2) + (n2)*(n3))

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

	double timing = timing_delay(&start, &end);
	uint64_t total_flop = flop_cublas + flop_atlas;

	fprintf(stderr, "Computation took %2.2f ms\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);

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

	flop_cublas += BLAS3_FLOP(nxC, nyC, nxA);
	ls_cublas += BLAS3_LS(nxC, nyC, nxA);

	release_data(descr->subA, 0);
	release_data(descr->subB, 0);
	release_data(descr->subC, 1<<0);
}
#endif

void core_mult(void *arg)
{
	COMMON_CODE

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nyC, nxC, nxA,
			 1.0f,  subA, ldA, subB, ldB, 0.0f, subC, ldC);

	flop_atlas += BLAS3_FLOP(nxC, nyC, nxA);
	ls_atlas += BLAS3_LS(nxC, nyC, nxA);

	release_data(descr->subA, 0);
	release_data(descr->subB, 0);
	release_data(descr->subC, 0);
}

unsigned nslicesx = 4;
unsigned nslicesy = 4;
unsigned xdim = 4096;
unsigned ydim = 4096;
unsigned zdim = 4096;
unsigned norandom = 0;

void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			nslicesy = nslicesx;
		}

		if (strcmp(argv[i], "-nblocksx") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocksy") == 0) {
			char *argptr;
			nslicesy = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-x") == 0) {
			char *argptr;
			xdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-y") == 0) {
			char *argptr;
			ydim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-z") == 0) {
			char *argptr;
			zdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-no-random") == 0) {
			norandom = 1;
		}
	}
}

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned i,j;

	int jobcounter;

	parse_args(argc, argv);

	x = xdim;
	y = ydim;
	z = zdim;

	jobcounter = nslicesx * nslicesy;

	A = malloc(z*y*sizeof(float));
	B = malloc(x*z*sizeof(float));
	C = malloc(x*y*sizeof(float));

	/* fill the A and B matrices */
	if (norandom) {
		for (i=0; i < z; i++) {
			for (j=0; j < y; j++) {
				A[i+j*z] = (float)(i);
			}
		}
	
		for (i=0; i < x; i++) {
			for (j=0; j < z; j++) {
				B[i+j*x] = (float)(j);
			}
		}
	} 
	else {
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
	}
	for (i=0; i < x; i++) {
		for (j=0; j < y; j++) {
			C[i+j*x] = (float)(0);
		}
	}

	/* start the runtime */
	init_machine();
	init_workers();

#ifdef USE_FXT
	fxt_register_thread(0);
#endif

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
	partition_data(&A_state, &f2);

	map_filters(&C_state, 2, &f, &f2);

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

			jobarguments[task].subA = get_sub_data(&A_state, 1, tasky);
			jobarguments[task].subB = get_sub_data(&B_state, 1, taskx);
			jobarguments[task].subC = get_sub_data(&C_state, 2, taskx, tasky);

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
