#include <examples/mult/dw_mult.h>

float *A, *B, *C;
data_state A_state, B_state, C_state;

/*
 * That program should compute C = A * B 
 * 
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)

              |---------------|
            z |       B       |
              |---------------|
       z              x
     |----|   |---------------|
     |    |   |               |
     |    |   |               |
     | A  | y |       C       |
     |    |   |               |
     |    |   |               |
     |----|   |---------------|

 */

void terminate(void)
{

	fprintf(stderr, "unpartition !!\n");
	unpartition_data(&C_state, 0);

	delete_data(&C_state);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	uint64_t total_flop = BLAS3_FLOP(ydim, xdim, zdim);
	uint64_t total_ls = ls_cublas + ls_atlas;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);
	fprintf(stderr, "	GB : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_ls/1000000000.0f, (double)ls_cublas/1000000000.0f, (double)ls_atlas/1000000000.0f);
	fprintf(stderr, "	GB/s : %2.2f\n", (double)total_ls / (double)timing/1000);

#ifdef CHECK_OUTPUT
	/* check results */
	/* compute C = C - AB */

	SGEMM("N", "N", ydim, xdim, zdim, -1.0f, A, ydim, B, zdim, 1.0f, C, ydim);
		
	/* make sure C = 0 */
	float err;
	err = SASUM(xdim*ydim, C, 1);	
	
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
	uint32_t nxC, nyC, nyA;		\
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
	nyA = descr[0].blas.ny;		\
					\
	ldA = descr[0].blas.ld;		\
	ldB = descr[1].blas.ld;		\
	ldC = descr[2].blas.ld;



#ifdef USE_CUDA
void cublas_mult(data_interface_t *descr, __attribute__((unused)) void *arg)
{
	COMMON_CODE

	tick_t sgemm_start;
	tick_t sgemm_end;


	GET_TICK(sgemm_start);

	cublasSgemm('n', 'n', nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 
					     0.0f, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		CUBLAS_REPORT_ERROR(st);

	GET_TICK(sgemm_end);

	uint64_t flopcnt = BLAS3_FLOP(nyC, nxC, nyA);

	flop_cublas += flopcnt;
	ls_cublas += BLAS3_LS(nyC, nxC, nyA);
}
#endif

void core_mult(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	COMMON_CODE

	SGEMM("N", "N", nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 0.0f, subC, ldC);

	flop_atlas += BLAS3_FLOP(nxC, nyC, nyA);
	ls_atlas += BLAS3_LS(nxC, nyC, nyA);
}

static void init_problem_data(void)
{
	unsigned i,j;

#ifdef USE_CUDA
	if (pin) {
		malloc_pinned_if_possible(&A, zdim*ydim*sizeof(float));
		malloc_pinned_if_possible(&B, xdim*zdim*sizeof(float));
		malloc_pinned_if_possible(&C, xdim*ydim*sizeof(float));
	} else
#endif
	{
		posix_memalign((void **)&A, 4096, zdim*ydim*sizeof(float));
		posix_memalign((void **)&B, 4096, xdim*zdim*sizeof(float));
		posix_memalign((void **)&C, 4096, xdim*ydim*sizeof(float));
	}

	/* fill the A and B matrices */
	if (norandom) {
		for (j=0; j < ydim; j++) {
			for (i=0; i < zdim; i++) {
				A[j+i*ydim] = (float)(i);
			}
		}
	
		for (j=0; j < zdim; j++) {
			for (i=0; i < xdim; i++) {
				B[j+i*zdim] = (float)(j);
			}
		}
	} 
	else {
#ifdef NORANDOM
		srand(2008);
		STARPU_ASSERT(0);
#endif
		for (j=0; j < ydim; j++) {
			for (i=0; i < zdim; i++) {
				A[j+i*ydim] = (float)(drand48());
			}
		}
	
		for (j=0; j < zdim; j++) {
			for (i=0; i < xdim; i++) {
				B[j+i*zdim] = (float)(drand48());
			}
		}
	}

	for (j=0; j < ydim; j++) {
		for (i=0; i < xdim; i++) {
			C[j+i*ydim] = (float)(0);
		}
	}
}

static void partition_mult_data(void)
{
	gettimeofday(&start, NULL);

	monitor_blas_data(&A_state, 0, (uintptr_t)A, 
		ydim, ydim, zdim, sizeof(float));
	monitor_blas_data(&B_state, 0, (uintptr_t)B, 
		zdim, zdim, xdim, sizeof(float));
	monitor_blas_data(&C_state, 0, (uintptr_t)C, 
		ydim, ydim, xdim, sizeof(float));

	conf.k = zdim;
	conf.m = ydim/nslicesy;
	conf.n = xdim/nslicesx;

	filter f;
	f.filter_func = vertical_block_filter_func;
	f.filter_arg = nslicesx;
		
	filter f2;
	f2.filter_func = block_filter_func;
	f2.filter_arg = nslicesy;
		
	partition_data(&B_state, &f);
	partition_data(&A_state, &f2);

	map_filters(&C_state, 2, &f, &f2);
}

static void launch_codelets(void)
{
#ifdef USE_FXT
	fxt_register_thread(0);
#endif
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
			jb = job_create();

			jb->where = CORE;

			cl->cl_arg = &conf;
			cl->cl_arg_size = sizeof(struct block_conf);
			cl->core_func = core_mult;
#ifdef USE_CUDA
			jb->where |= CUBLAS;
			cl->cublas_func = cublas_mult;
#endif
#ifdef USE_GORDON
			jb->where |= GORDON;
			cl->gordon_func = SPU_FUNC_SGEMM;
#endif
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
			jb->buffers[2].mode = RW;

			if (use_common_model)
			{
				jb->model = &sgemm_model_common;
			}
			else
			{
				jb->model = &sgemm_model;
			}
			
			push_task(jb);

		}
	}
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	init_machine();

	sem_init(&sem, 0, 0U);

	init_problem_data();

	partition_mult_data();

	launch_codelets();

	sem_wait(&sem);
	sem_destroy(&sem);

	terminate_machine();

	return 0;
}
