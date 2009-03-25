#include <examples/mult/dw_mult.h>

struct pos {
	unsigned x,y, z,iter;
};

struct pos currentpos [MAXSLICESY][MAXSLICESX];

float *A[MAXSLICESY][MAXSLICESZ];
float *B[MAXSLICESZ][MAXSLICESX];
float *C[MAXSLICESY][MAXSLICESX];

data_handle A_state[MAXSLICESY][MAXSLICESZ];
data_handle B_state[MAXSLICESZ][MAXSLICESX];
data_handle C_state[MAXSLICESY][MAXSLICESX];


static void callback_func_3(void *arg);
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

static codelet *cl;

static void terminate(void)
{
	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	uint64_t total_flop = BLAS3_FLOP(ydim, xdim, zdim)*niter;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);

	sem_post(&sem);
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
	memset(&A_state, 0, MAXSLICESY*MAXSLICESZ*sizeof(data_handle));
	memset(&B_state, 0, MAXSLICESZ*MAXSLICESZ*sizeof(data_handle));
	memset(&C_state, 0, MAXSLICESY*MAXSLICESX*sizeof(data_handle));

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
			currentpos[y][x].x = x;
			currentpos[y][x].y = y;
			currentpos[y][x].z = 0;
			currentpos[y][x].iter = 0;
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
					C[blocky][blockx][i*BLOCKSIZEY + j] = (float)0;
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

	display_memory_consumption();
}

static void cleanup_problem(void)
{
	unsigned z, y, x;

#ifdef CHECK_OUTPUT
	float maxerr = 0.0;
	float err;
	fprintf(stderr, "Checking results ....");

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			for (z = 0; z < nslicesz; z++)
			{
				SGEMM("N", "N", BLOCKSIZEY, BLOCKSIZEX, BLOCKSIZEZ, -(float)(niter), A[y][z], BLOCKSIZEY, B[z][x], BLOCKSIZEZ, 1.0f, C[y][x], BLOCKSIZEY);

			}

			/* make sure C - niter AB = 0 */
			err = SASUM(BLOCKSIZEX*BLOCKSIZEY, C[y][x], 1);

			if (err > BLOCKSIZEX*BLOCKSIZEY*niter*0.001) 
				fprintf(stderr, "\nerr = %f ( x = %d y = %d ) ... ", err/niter, x, y );

			maxerr = MAX(err, maxerr);
		}
	}

	if (maxerr > BLOCKSIZEX*BLOCKSIZEY*niter*0.001)
	{
		fprintf(stderr, " maxerr = %f\n", maxerr/niter);
	}
	else {
		fprintf(stderr, " OK\n");
	}
	fflush(stderr);
#endif

	for (y = 0; y < nslicesy; y++)
	{
		for (z = 0; z < nslicesz; z++)
		{
	//		free(A[y][z]);
		}
	}

	for (z = 0; z < nslicesz; z++)
	{
		for (x = 0; x < nslicesx; x++)
		{
	//		free(B[z][x]);
		}
	}

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
	//		free(C[y][x]);
		}
	}

	
	
}

int xycounter;

struct cb2_s {
	unsigned blockx;
	unsigned blocky;
	unsigned iter;
	int *xycounter;
};


static void construct_job(unsigned x, unsigned y, unsigned z, unsigned iter, struct pos *posp)
{
	job_t jb;
	jb = job_create();

	jb->cl = cl;

	jb->nbuffers = 3;

	jb->buffers[0].state = A_state[y][z];
	jb->buffers[0].mode = R;
	jb->buffers[1].state = B_state[z][x];
	jb->buffers[1].mode = R;
	jb->buffers[2].state = C_state[y][x];
	jb->buffers[2].mode = RW;

	jb->cb = callback_func_3;
	jb->argcb = posp;

	posp->z = z;
	posp->iter = iter;

	submit_job(jb);
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

static void callback_func_3(void *arg)
{
	/* the argument is a pointer to a counter of the remaining jobs */
	struct pos *posp = arg;
	unsigned x,y,z,iter;

	iter = posp->iter;
	x = posp->x;
	y = posp->y;
	z = posp->z;

	if (z < nslicesz - 1)
	{
		construct_job(x, y, z+1, iter, posp);
	}
	else
	{
		if (iter < niter - 1)
		{
			construct_job(x, y, 0, iter+1, posp);
		}
		else
		{
			callback_func(&xycounter);
		}
	}
}




static void launch_codelets(void)
{
#ifdef USE_FXT
	fxt_register_thread(0);
#endif
	/* partition the work into slices */
	unsigned taskx, tasky;

	/* only a callback per (nslicesz * niter) task given deps */
	xycounter = nslicesx * nslicesy;

	srand(time(NULL));

	gettimeofday(&start, NULL);

	for (taskx = 0; taskx < nslicesx; taskx++) 
	for (tasky = 0; tasky < nslicesy; tasky++)
	{
		construct_job(taskx, tasky, 0, 0, &currentpos[tasky][taskx]);
	}
}

static void init_codelet(void)
{
	cl = malloc(sizeof(codelet));

	cl->cl_arg = &conf;
	cl->cl_arg_size = sizeof(struct block_conf);
	cl->core_func = core_mult;
#ifdef USE_CUDA
	cl->cublas_func = cublas_mult;
#endif
#ifdef USE_GORDON
	cl->gordon_func = SPU_FUNC_SGEMM;
#endif

	cl->where = CORE;
#ifdef USE_CUDA
	cl->where |= CUBLAS;
#endif
#ifdef USE_GORDON
	cl->where |= GORDON;
#endif


}


int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	init_machine();

	sem_init(&sem, 0, 0U);

	init_problem_data();

	init_codelet();

	launch_codelets();

	sem_wait(&sem);
	sem_destroy(&sem);

	cleanup_problem();

	exit(-1);
	terminate_machine();

	return 0;
}
