#include <examples/mult/dw_mult.h>

float *A[MAXSLICESY][MAXSLICESZ];
float *B[MAXSLICESZ][MAXSLICESX];
float *C[MAXSLICESY][MAXSLICESX];

data_state A_state[MAXSLICESY][MAXSLICESZ];
data_state B_state[MAXSLICESZ][MAXSLICESX];
data_state C_state[MAXSLICESY][MAXSLICESX];

/* fortran ordering ... */
#define FULLA(i,j)	\
	(A[(i)/BLOCKSIZEY][(j)/BLOCKSIZEZ][(i)%BLOCKSIZEY + ((j)%BLOCKSIZEZ)*BLOCKSIZEY])

#define FULLB(i,j)	\
	(B[(i)/BLOCKSIZEZ][(j)/BLOCKSIZEX][(i)%BLOCKSIZEZ + ((j)%BLOCKSIZEX)*BLOCKSIZEZ])

#define FULLC(i,j)	\
	(C[(i)/BLOCKSIZEY][(j)/BLOCKSIZEX][(i)%BLOCKSIZEY + ((j)%BLOCKSIZEX)*BLOCKSIZEY])

#define TAG(x,y,z,iter)	\
		((x) + (y)*nslicesx + (z)*nslicesx*nslicesy + (iter)*nslicesx*nslicesy*nslicesz)

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

static void terminate(void)
{
	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	uint64_t total_flop = BLAS3_FLOP(ydim, xdim, zdim)*niter;
	uint64_t total_ls = ls_cublas + ls_atlas;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);

	sem_post(&sem);
}

static void callback_func(void *arg)
{
	/* the argument is a pointer to a counter of the remaining jobs */
	int *counter = arg;
	int newvalue = ATOMIC_ADD(counter, -1);
	if (newvalue == 0)
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
static void cublas_mult(data_interface_t *descr, __attribute__((unused)) void *arg)
{
	COMMON_CODE

	tick_t sgemm_start;
	tick_t sgemm_end;

	GET_TICK(sgemm_start);

	cublasSgemm('n', 'n', nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 
					     1.0f, subC, ldC);
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

static void core_mult(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	COMMON_CODE

//	fprintf(stderr, "Call SGEMM : nxC %d nyC %d nyA %d subA %p ldA %d subB %p ldB %d subC %p ldC %d\n",
//				nxC, nyC, nyA, subA, ldA, subB, ldB, subC, ldC);
	SGEMM("N", "N", nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 1.0f, subC, ldC);

	flop_atlas += BLAS3_FLOP(nxC, nyC, nyA);
	ls_atlas += BLAS3_LS(nxC, nyC, nyA);
}

#define MEM_ALIGNMENT	16

static void init_problem_data(void)
{
	unsigned i,j;

	/* debug ... */
	memset(A, 0, MAXSLICESY*MAXSLICESZ*sizeof(float *));
	memset(B, 0, MAXSLICESZ*MAXSLICESZ*sizeof(float *));
	memset(C, 0, MAXSLICESY*MAXSLICESX*sizeof(float *));
	memset(&A_state, 0, MAXSLICESY*MAXSLICESZ*sizeof(data_state));
	memset(&B_state, 0, MAXSLICESZ*MAXSLICESZ*sizeof(data_state));
	memset(&C_state, 0, MAXSLICESY*MAXSLICESX*sizeof(data_state));



	/* Allocate grids of buffer */
	/* TODO pin ... */
	unsigned z, y, x;

	for (y = 0; y < nslicesy; y++)
	{
		for (z = 0; z < nslicesz; z++)
		{
			posix_memalign((void **)&A[y][z], MEM_ALIGNMENT, BLOCKSIZEZ*BLOCKSIZEY*sizeof(float));
			assert(A[y][z]);
		}
	}

	for (z = 0; z < nslicesz; z++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			posix_memalign((void **)&B[z][x], MEM_ALIGNMENT, BLOCKSIZEX*BLOCKSIZEZ*sizeof(float));
			assert(B[z][x]);
		}
	}

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			posix_memalign((void **)&C[y][x], MEM_ALIGNMENT, BLOCKSIZEX*BLOCKSIZEY*sizeof(float));
			assert(C[y][x]);
		}
	}
	
	/* fill the A and B matrices */
	unsigned blockx, blocky, blockz;

	if (norandom) {
		for (blocky = 0; blocky < nslicesy; blocky++)
			for (blockz = 0; blockz < nslicesz; blockz++)
				for (j = 0; j < BLOCKSIZEY; j++)
					for (i = 0; i < BLOCKSIZEZ; i++)
					{
						A[blocky][blockz][i*BLOCKSIZEY + j] = (float)(1 + blockz + blocky*nslicesz);
					}

		for (blockz = 0; blockz < nslicesz; blockz++)
			for (blockx = 0; blockx < nslicesx; blockx++)
				for (j = 0; j < BLOCKSIZEZ; j++)
					for (i = 0; i < BLOCKSIZEX; i++)
					{
						B[blockz][blockx][i*BLOCKSIZEZ + j] = (float)(1 + blockx + blockz*nslicesx);
					}
	} 
	else {
		for (blocky = 0; blocky < nslicesy; blocky++)
			for (blockz = 0; blockz < nslicesz; blockz++)
				for (j = 0; j < BLOCKSIZEY; j++)
					for (i = 0; i < BLOCKSIZEZ; i++)
					{
						A[blocky][blockz][i*BLOCKSIZEY + j] = (float)(drand48());
					}

		for (blockz = 0; blockz < nslicesz; blockz++)
			for (blockx = 0; blockx < nslicesx; blockx++)
				for (j = 0; j < BLOCKSIZEZ; j++)
					for (i = 0; i < BLOCKSIZEX; i++)
					{
						B[blockz][blockx][i*BLOCKSIZEZ + j] = (float)(drand48());
					}

	}

	for (blocky = 0; blocky < nslicesy; blocky++)
		for (blockx = 0; blockx < nslicesx; blockx++)
			for (j = 0; j < BLOCKSIZEY; j++)
				for (i = 0; i < BLOCKSIZEX; i++)
				{
					C[blocky][blockx][i*BLOCKSIZEY + j] = (float)(blockx + blocky*nslicesx + 1);
				}


	/* declare the StarPU data to monitor */
	for (y = 0; y < nslicesy; y++)
	{
		for (z = 0; z < nslicesz; z++)
		{
			monitor_blas_data(&A_state[y][z], 0, (uintptr_t)A[y][z], 
				BLOCKSIZEY, BLOCKSIZEY, BLOCKSIZEZ, sizeof(float));
		}
	}

	for (z = 0; z < nslicesz; z++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			monitor_blas_data(&B_state[z][x], 0, (uintptr_t)B[z][x], 
				BLOCKSIZEZ, BLOCKSIZEZ, BLOCKSIZEX, sizeof(float));
		}
	}

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			monitor_blas_data(&C_state[y][x], 0, (uintptr_t)C[y][x], 
				BLOCKSIZEY, BLOCKSIZEY, BLOCKSIZEX, sizeof(float));
		}
	}



	conf.k = BLOCKSIZEZ;
	conf.m = BLOCKSIZEY;
	conf.n = BLOCKSIZEX;

	gettimeofday(&start, NULL);

	fprintf(stderr, "Memory consumption : A -> %ld KB\n", (MAXSLICESY*MAXSLICESZ*sizeof(float *))/(1024));
	fprintf(stderr, "Memory consumption : B -> %ld KB\n", (MAXSLICESZ*MAXSLICESX*sizeof(float *))/(1024));
	fprintf(stderr, "Memory consumption : C -> %ld KB\n", (MAXSLICESY*MAXSLICESX*sizeof(float *))/(1024));
	fprintf(stderr, "Memory consumption : A_state -> %ld KB\n", (MAXSLICESY*MAXSLICESZ*sizeof(data_state)/(1024)));
	fprintf(stderr, "Memory consumption : B_state -> %ld KB\n", (MAXSLICESZ*MAXSLICESX*sizeof(data_state)/(1024)));
	fprintf(stderr, "Memory consumption : C_state -> %ld KB\n", (MAXSLICESY*MAXSLICESX*sizeof(data_state)/(1024)));
	fprintf(stderr, "Memory consumption : data A -> %ld MB\n", (ydim*zdim*sizeof(float)/(1024*1024)));
	fprintf(stderr, "Memory consumption : data B -> %ld MB\n", (zdim*xdim*sizeof(float)/(1024*1024)));
	fprintf(stderr, "Memory consumption : data C -> %ld MB\n", (ydim*xdim*sizeof(float)/(1024*1024)));

	fprintf(stderr, "Total memory : %ld MB\n", (MAXSLICESY*MAXSLICESZ*sizeof(float *) + MAXSLICESZ*MAXSLICESX*sizeof(float *) + MAXSLICESY*MAXSLICESX*sizeof(float *) + MAXSLICESY*MAXSLICESZ*sizeof(data_state) + MAXSLICESZ*MAXSLICESX*sizeof(data_state) + MAXSLICESY*MAXSLICESX*sizeof(data_state) + ydim*zdim*sizeof(float) +  zdim*xdim*sizeof(float) +  ydim*xdim*sizeof(float))/(1024*1024) );

}

static void cleanup_problem(void)
{
	unsigned z, y, x;

	for (y = 0; y < nslicesy; y++)
	{
		for (z = 0; z < nslicesz; z++)
		{
			free(A[y][z]);
		}
	}

	for (z = 0; z < nslicesz; z++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			free(B[z][x]);
		}
	}

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			free(C[y][x]);
		}
	}
	
}


static void launch_codelets(void)
{
#ifdef USE_FXT
	fxt_register_thread(0);
#endif
	/* partition the work into slices */
	unsigned taskx, tasky, taskz;
	job_t jb;

	/* only a callback per (nslicesz * niter) task given deps */
	jobcounter = nslicesx * nslicesy;

	srand(time(NULL));

	codelet *cl = malloc(sizeof(codelet));

	cl->cl_arg = &conf;
	cl->cl_arg_size = sizeof(struct block_conf);
	cl->core_func = core_mult;
#ifdef USE_CUDA
	cl->cublas_func = cublas_mult;
#endif
#ifdef USE_GORDON
	cl->gordon_func = SPU_FUNC_SGEMM;
#endif
	unsigned iter;

	for (iter = 0; iter < niter; iter++)
	for (taskz = 0; taskz < nslicesz; taskz++) 
	for (taskx = 0; taskx < nslicesx; taskx++) 
	for (tasky = 0; tasky < nslicesy; tasky++)
	{
//		fprintf(stderr, "TASK X %d Y %d Z %d\n", taskx, tasky, taskz);

		/* A B[task] = C[task] */
		jb = job_create();

		jb->where = CORE;
#ifdef USE_CUDA
		jb->where |= CUBLAS;
#endif
#ifdef USE_GORDON
		jb->where |= GORDON;
#endif

		jb->cl = cl;

		tag_t tag = TAG(taskz, tasky, taskx, iter);
		//fprintf(stderr, "DECLARE TAG(taskz %d, tasky %d, taskx %d, iter %d) %lx\n", taskz, tasky, taskx, iter, tag);
		jb->nbuffers = 3;

		jb->buffers[0].state = &A_state[tasky][taskz];
		jb->buffers[0].mode = R;
		jb->buffers[1].state = &B_state[taskz][taskx];
		jb->buffers[1].mode = R;
		jb->buffers[2].state = &C_state[tasky][taskx];
		jb->buffers[2].mode = RW;

		jb->model = &sgemm_model;

		tag_declare(tag, jb);

		if ((taskz == 0) && (iter == 0))
		{	
			jb->cb = callback_func;
			jb->argcb = &jobcounter;
		}

		if (taskz < nslicesz - 1)
		{
			tag_declare_deps(TAG(taskz, tasky, taskx, iter), 1, TAG(taskz+1, tasky, taskx, iter));
		}
		else if (iter < niter - 1) {
			tag_declare_deps(TAG(taskz, tasky, taskx, iter), 1, TAG(0, tasky, taskx, iter+1));
		} else {
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

	launch_codelets();

	sem_wait(&sem);
	sem_destroy(&sem);

	cleanup_problem();

	exit(-1);
	terminate_machine();

	return 0;
}
