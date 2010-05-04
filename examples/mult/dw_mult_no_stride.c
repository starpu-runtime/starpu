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
#ifdef STARPU_USE_GORDON
#include "gordon/func_sgemm_ibm.h"
#endif

float *A[MAXSLICESY][MAXSLICESZ];
float *B[MAXSLICESZ][MAXSLICESX];
float *C[MAXSLICESY][MAXSLICESX];

starpu_data_handle A_state[MAXSLICESY][MAXSLICESZ];
starpu_data_handle B_state[MAXSLICESZ][MAXSLICESX];
starpu_data_handle C_state[MAXSLICESY][MAXSLICESX];

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

#define MEM_ALIGNMENT	16

static void init_problem_data(void)
{
	unsigned i,j;

	/* Allocate grids of buffer */
	/* TODO pin ... */
	unsigned z, y, x;

	for (y = 0; y < nslicesy; y++)
	{
		for (z = 0; z < nslicesz; z++)
		{
#ifdef STARPU_HAVE_POSIX_MEMALIGN
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
#ifdef STARPU_HAVE_POSIX_MEMALIGN
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
#ifdef STARPU_HAVE_POSIX_MEMALIGN
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
						A[blocky][blockz][i*BLOCKSIZEY + j] = (float)(starpu_drand48());
					}

		for (blockz = 0; blockz < nslicesz; blockz++)
			for (blockx = 0; blockx < nslicesx; blockx++)
				for (j = 0; j < BLOCKSIZEZ; j++)
					for (i = 0; i < BLOCKSIZEX; i++)
					{
						B[blockz][blockx][i*BLOCKSIZEZ + j] = (float)(starpu_drand48());
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
			starpu_register_matrix_data(&A_state[y][z], 0, (uintptr_t)A[y][z], 
				BLOCKSIZEY, BLOCKSIZEY, BLOCKSIZEZ, sizeof(float));
		}
	}

	for (z = 0; z < nslicesz; z++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			starpu_register_matrix_data(&B_state[z][x], 0, (uintptr_t)B[z][x], 
				BLOCKSIZEZ, BLOCKSIZEZ, BLOCKSIZEX, sizeof(float));
		}
	}

	for (y = 0; y < nslicesy; y++)
	{
		for (x = 0; x < nslicesx; x++)
		{
			starpu_register_matrix_data(&C_state[y][x], 0, (uintptr_t)C[y][x], 
				BLOCKSIZEY, BLOCKSIZEY, BLOCKSIZEX, sizeof(float));
		}
	}

#ifdef STARPU_USE_GORDON
	conf.k = BLOCKSIZEZ;
	conf.m = BLOCKSIZEY;
	conf.n = BLOCKSIZEX;
#endif

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

struct cb2_s {
	unsigned blockx;
	unsigned blocky;
	unsigned iter;
};

static starpu_codelet cl = {
	.cpu_func = cpu_mult,
#ifdef STARPU_USE_CUDA
	.cuda_func = cublas_mult,
#endif
#ifdef STARPU_USE_GORDON
	/* .gordon_func will be set by load_elf_sgemm */
#endif

	.model = &sgemm_model,
	.where = STARPU_CPU|STARPU_CUDA|STARPU_GORDON,
	.nbuffers = 3
};

#ifdef STARPU_USE_GORDON
static const char *spu_func_sgemm_elf_file = "./gordon/func_sgemm_ibm.spuelf";
static unsigned spu_func_sgemm_elf_id;
static unsigned spu_func_sgemm_ibm_id;

static void load_elf_sgemm(void)
{
	spu_func_sgemm_elf_id =
		gordon_register_elf_plugin(spu_func_sgemm_elf_file);

	spu_func_sgemm_ibm_id = gordon_register_kernel(spu_func_sgemm_elf_id, "func_sgemm_ibm");

	
	gordon_load_plugin_on_all_spu(spu_func_sgemm_elf_id);
	gordon_load_kernel_on_all_spu(spu_func_sgemm_ibm_id);

	cl.gordon_func = spu_func_sgemm_ibm_id;
}

#endif // STARPU_USE_GORDON

static struct starpu_task *construct_task(unsigned x, unsigned y, unsigned z, unsigned iter)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl;

#ifdef STARPU_USE_GORDON
	task->cl_arg = &conf;
	task->cl_arg_size = sizeof(struct ibm_sgemm_block_conf);
#endif

	task->use_tag = 1;
	task->tag_id = TAG(z, y, x, iter);

	task->buffers[0].handle = A_state[y][z];
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = B_state[z][x];
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = C_state[y][x];
	task->buffers[2].mode = STARPU_RW;

	return task;
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

	/* do some accounting */
	int id = starpu_get_worker_id();
	flop_per_worker[id] += BLAS3_FLOP(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
	ls_per_worker[id] += BLAS3_LS(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);

	/* TAG(nslicesz - 1, y, x, iter) remains ... */
	for (z = 0; z < nslicesz - 1; z++)
	{
		starpu_tag_remove(TAG(z, y, x, iter));
	}

	if (iter > 0)
	{
		starpu_tag_remove(TAG(nslicesz - 1, y, x, iter-1));
	}
	
	if (iter != niter - 1) {
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
			task->callback_func = callback_func_2;
			task->callback_arg = cb2;
		}

		starpu_task_submit(task);
	}
}

static void launch_codelets(void)
{
#ifdef STARPU_USE_FXT
	_starpu_fxt_register_thread(0);
#endif
	/* partition the work into slices */
	unsigned taskx, tasky;

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

	starpu_helper_init_cublas();

#ifdef STARPU_USE_GORDON
	load_elf_sgemm();
#endif

	init_problem_data();

	launch_codelets();

	starpu_wait_all_tasks();

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	display_stats(timing);


	cleanup_problem();

	starpu_helper_shutdown_cublas();
	starpu_shutdown();

	return 0;
}
