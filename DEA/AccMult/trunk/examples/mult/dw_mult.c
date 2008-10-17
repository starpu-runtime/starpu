#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
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
#include <datawizard/interfaces/blas_interface.h>
#include <datawizard/interfaces/blas_filters.h>

#include <task-models/blas_model.h>

#include <common/fxt.h>

#if defined (USE_CUBLAS) || defined (USE_CUDA)
#include <cuda.h>
#endif

extern struct perfmodel_t sgemm_model;

sem_t sem;

float *A;
float *B;
float *C;

data_state A_state;
data_state B_state;
data_state C_state;

tick_t start;
tick_t end;

unsigned nslicesx = 4;
unsigned nslicesy = 4;
unsigned xdim = 4096;
unsigned ydim = 4096;
unsigned zdim = 4096;
unsigned norandom = 0;
unsigned pin = 0;

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

	fprintf(stderr, "unpartition !!\n");
	unpartition_data(&C_state, 0);

	delete_data(&C_state);

	GET_TICK(end);

	double timing = timing_delay(&start, &end);
	uint64_t total_flop = flop_cublas + flop_atlas;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);

#ifdef CHECK_OUTPUT
	/* check results */
	/* compute C = C - AB */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ydim, xdim, zdim,
			 -1.0f,  A, zdim, B, xdim, 1.0f, C, xdim);
		
	/* make sure C = 0 */
	float err;
	err = cblas_sasum(xdim*ydim, C, 1);	
	
	if (err < xdim*ydim*0.001) {
		fprintf(stderr, "Results are OK\n");
	}
	else {
		fprintf(stderr, "There were errors ... err = %f\n", err);
	}
#endif // CHECK_OUTPUT

	sem_post(&sem);
}

void callback_func(void *arg)
{
	/* the argument is a pointer to a counter of the remaining jobs */
	int *counter = arg;
	*counter -= 1;
	if (*counter == 0)
	{
		/* we are done */	
		fprintf(stderr, "done ...\n");
		terminate();
	}

	return;
}


#define COMMON_CODE			\
	uint32_t nxC, nyC, nxA;		\
	uint32_t ldA, ldB, ldC;		\
					\
	float *subA;			\
	float *subB;			\
	float *subC;			\
					\
	subA = (float *)descr[0].blas.ptr;	\
	subB = (float *)descr[1].blas.ptr;	\
	subC = (float *)descr[2].blas.ptr;	\
					\
	nxC = descr[2].blas.nx;		\
	nyC = descr[2].blas.ny;		\
	nxA = descr[0].blas.nx;		\
						\
	ldA = descr[0].blas.ld;		\
	ldB = descr[1].blas.ld;		\
	ldC = descr[2].blas.ld;



#if defined (USE_CUBLAS) || defined (USE_CUDA)
void cublas_mult(data_interface_t *descr, __attribute__((unused)) void *arg)
{
	COMMON_CODE

	tick_t sgemm_start;
	tick_t sgemm_end;


	GET_TICK(sgemm_start);

	cublasSgemm('n', 'n', nxC, nyC, nxA, 1.0f, subB, ldB, subA, ldA, 
					     0.0f, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		CUBLAS_REPORT_ERROR(st);

	GET_TICK(sgemm_end);

	uint64_t flopcnt = BLAS3_FLOP(nxC, nyC, nxA);

	flop_cublas += flopcnt;
	ls_cublas += BLAS3_LS(nxC, nyC, nxA);
}
#endif

void core_mult(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	COMMON_CODE

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nyC, nxC, nxA,
			 1.0f,  subA, ldA, subB, ldB, 0.0f, subC, ldC);

	flop_atlas += BLAS3_FLOP(nxC, nyC, nxA);
	ls_atlas += BLAS3_LS(nxC, nyC, nxA);
}

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

		if (strcmp(argv[i], "-pin") == 0) {
			pin = 1;
		}
	}
}

/*
 * This is a codelet itself 
 */
void init_problem_codelet (__attribute__((unused)) buffer_descr *descr,
			   __attribute__((unused)) void *arg)
{
	unsigned i,j;

#if defined (USE_CUBLAS) || defined (USE_CUDA)
	if (pin) {
		cuMemAllocHost((void **)&A, zdim*ydim*sizeof(float));
		cuMemAllocHost((void **)&B, xdim*zdim*sizeof(float));
		cuMemAllocHost((void **)&C, xdim*ydim*sizeof(float));
	} else
#endif
	{
		A = malloc(zdim*ydim*sizeof(float));
		B = malloc(xdim*zdim*sizeof(float));
		C = malloc(xdim*ydim*sizeof(float));
	}

	/* fill the A and B matrices */
	if (norandom) {
		for (i=0; i < zdim; i++) {
			for (j=0; j < ydim; j++) {
				A[i+j*zdim] = (float)(i);
			}
		}
	
		for (i=0; i < xdim; i++) {
			for (j=0; j < zdim; j++) {
				B[i+j*xdim] = (float)(j);
			}
		}
	} 
	else {
#ifdef NORANDOM
		srand(2008);
		ASSERT(0);
#endif
		for (i=0; i < zdim; i++) {
			for (j=0; j < ydim; j++) {
				A[i+j*zdim] = (float)(drand48());
			}
		}
	
		for (i=0; i < xdim; i++) {
			for (j=0; j < zdim; j++) {
				B[i+j*xdim] = (float)(drand48());
			}
		}
	}
	for (i=0; i < xdim; i++) {
		for (j=0; j < ydim; j++) {
			C[i+j*xdim] = (float)(0);
		}
	}
}

int jobcounter;

void init_problem_callback(void *arg __attribute__((unused)))
{
#ifdef USE_FXT
	fxt_register_thread(0);
#endif

	GET_TICK(start);
	monitor_blas_data(&A_state, 0, (uintptr_t)A, 
		zdim, zdim, ydim, sizeof(float));
	monitor_blas_data(&B_state, 0, (uintptr_t)B, 
		xdim, xdim, zdim, sizeof(float));
	monitor_blas_data(&C_state, 0, (uintptr_t)C, 
		xdim, xdim, ydim, sizeof(float));

	filter f;
	f.filter_func = block_filter_func;
	f.filter_arg = nslicesx;
		
	filter f2;
	f2.filter_func = vertical_block_filter_func;
	f2.filter_arg = nslicesy;
		
	partition_data(&B_state, &f);
	partition_data(&A_state, &f2);

	map_filters(&C_state, 2, &f, &f2);

	/* partition the work into slices */
	unsigned taskx, tasky;
	job_t jb;

	jobcounter = nslicesx * nslicesy;

	srand(time(NULL));

	for (taskx = 0; taskx < nslicesx; taskx++) 
	{
		for (tasky = 0; tasky < nslicesy; tasky++)
		{
			/* A B[task] = C[task] */
			codelet *cl = malloc(sizeof(codelet));

			cl->cl_arg = NULL;
			cl->core_func = core_mult;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
			cl->cublas_func = cublas_mult;
#endif

			jb = job_create();
			jb->where = CORE | CUBLAS;
			jb->cb = callback_func;
			jb->argcb = &jobcounter;
			jb->cl = cl;

			tag_t tag = 
				((((unsigned long long)(taskx))<<32) 
				| (unsigned long long)(tasky));
			jb->nbuffers = 3;

			tag_declare(tag, jb);

			jb->buffers[0].state = get_sub_data(&A_state, 1, tasky);
			jb->buffers[0].mode = R;
			jb->buffers[1].state = get_sub_data(&B_state, 1, taskx);
			jb->buffers[1].mode = R;
			jb->buffers[2].state = 
				get_sub_data(&C_state, 2, taskx, tasky);
			jb->buffers[2].mode = W;

			jb->model = &sgemm_model;
			
			push_task(jb);
		}
	}
}

void init_problem(void)
{
	job_t jb;

	codelet *cl = malloc(sizeof(codelet));

	cl->cl_arg = NULL;
	cl->core_func = init_problem_codelet;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	cl->cublas_func = init_problem_codelet;
#endif

	jb = job_create();
	jb->type = CODELET;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	jb->where = CUBLAS;
#else
	jb->where = ANY;
#endif
	jb->cb = init_problem_callback;
	jb->cl = cl;

	push_task(jb);
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	init_machine();

	sem_init(&sem, 0, 0U);

	init_problem();
	sem_wait(&sem);
	sem_destroy(&sem);

	return 0;
}
