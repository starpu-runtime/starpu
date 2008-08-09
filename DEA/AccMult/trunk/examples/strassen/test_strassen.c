#include "strassen.h"

sem_t sem;

float *A;
float *B;
float *C;

data_state A_state;
data_state B_state;
data_state C_state;

tick_t start;
tick_t end;

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

void terminate(void *arg __attribute__ ((unused)))
{
	GET_TICK(end);

	double timing = timing_delay(&start, &end);
	uint64_t total_flop = flop_cublas + flop_atlas;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);

	sem_post(&sem);
}

unsigned xdim = 2048;
unsigned ydim = 2048;
unsigned zdim = 2048;
unsigned norandom = 0;
unsigned pin = 0;

void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
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
		srand(2008);
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

	strassen(&A_state, &B_state, &C_state, terminate, NULL);
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
