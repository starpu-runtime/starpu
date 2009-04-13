/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "dw_mult.h"

static pthread_mutex_t mutex;
static pthread_cond_t cond;

float *A[MAXSLICESY][MAXSLICESZ];
float *B[MAXSLICESZ][MAXSLICESX];
float *C[MAXSLICESY][MAXSLICESX];

starpu_data_handle A_state[MAXSLICESY][MAXSLICESZ];
starpu_data_handle B_state[MAXSLICESZ][MAXSLICESX];
starpu_data_handle C_state[MAXSLICESY][MAXSLICESX];

/* fortran ordering ... */
#define FULLA(i,j)	\
	(A[(i)/BLOCKSIZEY][(j)/BLOCKSIZEZ][(i)%BLOCKSIZEY + ((j)%BLOCKSIZEZ)*BLOCKSIZEY])

#define FULLB(i,j)	\
	(B[(i)/BLOCKSIZEZ][(j)/BLOCKSIZEX][(i)%BLOCKSIZEZ + ((j)%BLOCKSIZEX)*BLOCKSIZEZ])

#define FULLC(i,j)	\
	(C[(i)/BLOCKSIZEY][(j)/BLOCKSIZEX][(i)%BLOCKSIZEY + ((j)%BLOCKSIZEX)*BLOCKSIZEY])

#define TAG(x,y,z,iter)	\
		((starpu_tag_t)((z) + (iter)*nslicesz + (x)*(nslicesz*niter) + (y)*(nslicesx*nslicesz*niter)))

static void submit_new_iter(unsigned x, unsigned y, unsigned iter);

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

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);

	pthread_mutex_lock(&mutex);
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);
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
static void cublas_mult(starpu_data_interface_t *descr, __attribute__((unused)) void *arg)
{
	COMMON_CODE

	cublasSgemm('n', 'n', nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 
					     1.0f, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		STARPU_ASSERT(0);

	uint64_t flopcnt = BLAS3_FLOP(nyC, nxC, nyA);

	flop_cublas += flopcnt;
	ls_cublas += BLAS3_LS(nyC, nxC, nyA);
}
#endif

static void core_mult(starpu_data_interface_t *descr, __attribute__((unused))  void *arg)
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
	memset(&A_state, 0, MAXSLICESY*MAXSLICESZ*sizeof(starpu_data_handle));
	memset(&B_state, 0, MAXSLICESZ*MAXSLICESZ*sizeof(starpu_data_handle));
	memset(&C_state, 0, MAXSLICESY*MAXSLICESX*sizeof(starpu_data_handle));

	/* Allocate grids of buffer */
	/* TODO pin ... */
	unsigned z, y, x;

	for (y = 0; y < nslicesy; y++)
	{
		for (z = 0; z < nslicesz; z++)
		{
#ifdef HAVE_POSIX_MEMALIGN
			posix_memalign((void **)&A[y][z], MEM_ALIGNMENT, BLOCKSIZEZ*BLOCKSIZEY*sizeof(float));
#else
			A[y][z] = malloc(BLOCKSIZEZ*BLOCKSIZEY*sizeof(float));
#endif
			assert(A[y][z]);
		}
	}

	for (z = 0; z < nslicesz; z++)
	{
		for (x = 0; x < nslicesx; x++)
		{
#ifdef HAVE_POSIX_MEMALIGN
			posix_memalign((void **)&B[z][x], MEM_ALIGNMENT, BLOCKSIZEX*BLOCKSIZEZ*sizeof(float));
#else
			B[z][x] = malloc(BLOCKSIZEX*BLOCKSIZEZ*sizeof(float));
#endif
			assert(B[z][x]);
		}
	}

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
#ifdef HAVE_POSIX_MEMALIGN
			posix_memalign((void **)&C[y][x], MEM_ALIGNMENT, BLOCKSIZEX*BLOCKSIZEY*sizeof(float));
#else
			C[y][x] = malloc(BLOCKSIZEX*BLOCKSIZEY*sizeof(float));
#endif
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
			starpu_monitor_blas_data(&A_state[y][z], 0, (uintptr_t)A[y][z], 
				BLOCKSIZEY, BLOCKSIZEY, BLOCKSIZEZ, sizeof(float));
		}
	}

	for (z = 0; z < nslicesz; z++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			starpu_monitor_blas_data(&B_state[z][x], 0, (uintptr_t)B[z][x], 
				BLOCKSIZEZ, BLOCKSIZEZ, BLOCKSIZEX, sizeof(float));
		}
	}

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			starpu_monitor_blas_data(&C_state[y][x], 0, (uintptr_t)C[y][x], 
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
			starpu_tag_remove(TAG(nslicesz - 1, y, x, niter - 1));
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

static starpu_codelet cl = {
	.core_func = core_mult,
#ifdef USE_CUDA
	.cublas_func = cublas_mult,
#endif
#ifdef USE_GORDON
#ifdef SPU_FUNC_SGEMM
	.gordon_func = SPU_FUNC_SGEMM,
#else
#warning SPU_FUNC_SGEMM is not available
#endif
#endif
	.where = CORE|CUBLAS|GORDON,
	.nbuffers = 3
};

static struct starpu_task *construct_task(unsigned x, unsigned y, unsigned z, unsigned iter)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl;

	task->cl_arg = &conf;
	task->cl_arg_size = sizeof(struct block_conf);

	task->use_tag = 1;
	task->tag_id = TAG(z, y, x, iter);

	task->buffers[0].state = A_state[y][z];
	task->buffers[0].mode = R;
	task->buffers[1].state = B_state[z][x];
	task->buffers[1].mode = R;
	task->buffers[2].state = C_state[y][x];
	task->buffers[2].mode = RW;

	return task;
}


static void callback_func(void *arg)
{
	/* the argument is a pointer to a counter of the remaining tasks */
	int *counter = arg;
	int newvalue = STARPU_ATOMIC_ADD(counter, -1);
	if (newvalue == 0)
	{
		/* we are done */	
		fprintf(stderr, "done ...\n");
		terminate();
	}

	return;
}


static void callback_func_2(void *arg)
{
	/* the argument is a pointer to a counter of the remaining tasks */
	struct cb2_s *cb2 = arg;
	unsigned x,y,z,iter;

	iter = cb2->iter;
	x = cb2->blockx;
	y = cb2->blocky;

	free(cb2);

//	fprintf(stderr, "func 2 for x %d y %d iter %d\n", x, y, iter);

	/* TAG(nslicesz - 1, y, x, iter) remains ... */
	for (z = 0; z < nslicesz - 1; z++)
	{
		starpu_tag_remove(TAG(z, y, x, iter));
	}

	if (iter > 0)
	{
		starpu_tag_remove(TAG(nslicesz - 1, y, x, iter-1));
	}
	
	if (iter == niter - 1) {
		callback_func(&xycounter);
	}
	else {
		submit_new_iter(x, y, iter+1);
	}
}



static void submit_new_iter(unsigned x, unsigned y, unsigned iter)
{
	unsigned z;
	for (z = 0; z < nslicesz; z++) 
	{
		struct starpu_task *task;
		task = construct_task(x, y, z, iter);
		
		if (z != 0) {
			starpu_tag_declare_deps(TAG(z, y, x, iter), 1, TAG(z-1, y, x, iter));
		}

		if (z == nslicesz - 1) {
			struct cb2_s *cb2 = malloc(sizeof(struct cb2_s));
				cb2->blockx = x;
				cb2->blocky = y;
				cb2->iter = iter;
				cb2->xycounter = &xycounter;
			task->callback_func = callback_func_2;
			task->callback_arg = cb2;
		}

		starpu_submit_task(task);
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
		submit_new_iter(taskx, tasky, 0);
	}
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	starpu_init(NULL);

	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);

	init_problem_data();

	launch_codelets();

	pthread_mutex_lock(&mutex);
	pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	cleanup_problem();

	exit(-1);
	starpu_shutdown();

	return 0;
}
