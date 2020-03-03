/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/*
 * This computes an SPMV on a BCSR sparse matrix. It simply splits the matrix
 * into its blocks, thus turning the problem into mere matrix-vector products
 * (GEMV) which can be run in parallel.
 */
#include "dw_block_spmv.h"
#include "matrix_market/mm_to_bcsr.h"

#ifdef STARPU_HAVE_HELGRIND_H
#include <valgrind/helgrind.h>
#endif
#ifndef ANNOTATE_HAPPENS_BEFORE
#define ANNOTATE_HAPPENS_BEFORE(obj) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_AFTER
#define ANNOTATE_HAPPENS_AFTER(obj) ((void)0)
#endif

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

static double start;
static double end;

static sem_t sem;

static unsigned c = 256;
static unsigned r = 256;

static int remainingtasks = -1;

static starpu_data_handle_t sparse_matrix;
static starpu_data_handle_t vector_in, vector_out;

static uint32_t size;
static char *inputfile;
static bcsr_t *bcsr_matrix;

static float *vector_in_ptr;
static float *vector_out_ptr;

void create_data(void)
{
	/* read the input file */
	bcsr_matrix = mm_file_to_bcsr(inputfile, c, r);

	/* declare the corresponding block CSR to the runtime */
	starpu_bcsr_data_register(&sparse_matrix, STARPU_MAIN_RAM, bcsr_matrix->nnz_blocks, bcsr_matrix->nrows_blocks,
	                (uintptr_t)bcsr_matrix->val, bcsr_matrix->colind, bcsr_matrix->rowptr,
			0, bcsr_matrix->r, bcsr_matrix->c, sizeof(float));

	size = c*r*starpu_bcsr_get_nnz(sparse_matrix);
/*	printf("size = %d \n ", size); */

	/* initiate the 2 vectors */
	vector_in_ptr = malloc(size*sizeof(float));
	assert(vector_in_ptr);

	vector_out_ptr = malloc(size*sizeof(float));
	assert(vector_out_ptr);

	/* fill those */
	unsigned ind;
	for (ind = 0; ind < size; ind++)
	{
		vector_in_ptr[ind] = 2.0f;
		vector_out_ptr[ind] = 0.0f;
	}

	starpu_vector_data_register(&vector_in, STARPU_MAIN_RAM, (uintptr_t)vector_in_ptr, size, sizeof(float));
	starpu_vector_data_register(&vector_out, STARPU_MAIN_RAM, (uintptr_t)vector_out_ptr, size, sizeof(float));
}

void unregister_data(void)
{
	starpu_data_unpartition(sparse_matrix, STARPU_MAIN_RAM);
	starpu_data_unregister(sparse_matrix);

	starpu_data_unpartition(vector_in, STARPU_MAIN_RAM);
	starpu_data_unregister(vector_in);

	starpu_data_unpartition(vector_out, STARPU_MAIN_RAM);
	starpu_data_unregister(vector_out);
}

void init_problem_callback(void *arg)
{
	unsigned *remaining = arg;

	unsigned val = STARPU_ATOMIC_ADD(remaining, -1);
	ANNOTATE_HAPPENS_BEFORE(&remaining);

/*	if (val < 10)
		printf("callback %d remaining \n", val); */

	if ( val == 0 )
	{
		ANNOTATE_HAPPENS_AFTER(&remaining);
		printf("DONE ...\n");
		end = starpu_timing_now();

		sem_post(&sem);
	}
}

unsigned get_bcsr_nchildren(struct starpu_data_filter *f, starpu_data_handle_t handle)
{
  return (unsigned)starpu_bcsr_get_nnz(handle);
}

void call_filters(void)
{

	struct starpu_data_filter bcsr_f;
	struct starpu_data_filter vector_in_f, vector_out_f;

	bcsr_f.filter_func    = starpu_bcsr_filter_canonical_block;
	bcsr_f.get_nchildren = get_bcsr_nchildren;
	/* the children use a matrix interface ! */
	bcsr_f.get_child_ops = starpu_bcsr_filter_canonical_block_child_ops;

	vector_in_f.filter_func = starpu_vector_filter_block;
	vector_in_f.nchildren  = size/c;
	vector_in_f.get_nchildren  = NULL;
	vector_in_f.get_child_ops  = NULL;

	vector_out_f.filter_func = starpu_vector_filter_block;
	vector_out_f.nchildren  = size/r;
	vector_out_f.get_nchildren  = NULL;
	vector_out_f.get_child_ops  = NULL;

	starpu_data_partition(sparse_matrix, &bcsr_f);

	starpu_data_partition(vector_in, &vector_in_f);
	starpu_data_partition(vector_out, &vector_out_f);
}

#define NSPMV	32
unsigned totaltasks;

struct starpu_codelet cl =
{
	.cpu_funcs = { cpu_block_spmv},
	.cpu_funcs_name = { "cpu_block_spmv" },
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cublas_block_spmv},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW}
};

void launch_spmv_codelets(void)
{
	struct starpu_task *task_tab;
	uint8_t *is_entry_tab;

	/* we call one codelet per block */
	unsigned nblocks = starpu_bcsr_get_nnz(sparse_matrix);
	unsigned nrows = starpu_bcsr_get_nrow(sparse_matrix);

	remainingtasks = NSPMV*nblocks;
	totaltasks = remainingtasks;

	unsigned taskid = 0;

	task_tab = calloc(totaltasks, sizeof(struct starpu_task));
	STARPU_ASSERT(task_tab);

	is_entry_tab = calloc(totaltasks, sizeof(uint8_t));
	STARPU_ASSERT(is_entry_tab);

	printf("there will be %d codelets\n", remainingtasks);

	uint32_t *rowptr = starpu_bcsr_get_local_rowptr(sparse_matrix);
	uint32_t *colind = starpu_bcsr_get_local_colind(sparse_matrix);

	start = starpu_timing_now();

	unsigned loop;
	for (loop = 0; loop < NSPMV; loop++)
	{
		unsigned row;
		unsigned part = 0;

		for (row = 0; row < nrows; row++)
		{
			unsigned index;

			if (rowptr[row] == rowptr[row+1])
			{
				continue;
			}


			for (index = rowptr[row]; index < rowptr[row+1]; index++, part++)
			{
				struct starpu_task *task = &task_tab[taskid];
				starpu_task_init(task);

				task->use_tag = 1;
				task->tag_id = taskid;

				task->callback_func = init_problem_callback;
				task->callback_arg = &remainingtasks;
				task->cl = &cl;
				task->cl_arg = NULL;

				unsigned i = colind[index];
				unsigned j = row;

				task->handles[0] = starpu_data_get_sub_data(sparse_matrix, 1, part);
				task->handles[1] = starpu_data_get_sub_data(vector_in, 1, i);
				task->handles[2] = starpu_data_get_sub_data(vector_out, 1, j);

				/* all tasks in the same row are dependant so that we don't wait too much for data
				 * we need to wait on the previous task if we are not the first task of a row */
				if (index != rowptr[row & ~0x3])
				{
					/* this is not the first task in the row */
					starpu_tag_declare_deps((starpu_tag_t)taskid, 1, (starpu_tag_t)(taskid-1));

					is_entry_tab[taskid] = 0;
				}
				else
				{
					/* this is an entry task */
					is_entry_tab[taskid] = 1;
				}

				taskid++;
			}
		}
	}

	printf("start submitting tasks !\n");

	/* submit ALL tasks now */
	unsigned nchains = 0;
	unsigned task;
	for (task = 0; task < totaltasks; task++)
	{
		int ret;
		if (is_entry_tab[task])
		{
			nchains++;
		}

		ret = starpu_task_submit(&task_tab[task]);
		if (ret == -ENODEV)
			exit(77);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	printf("end of task submission (there was %u chains for %u tasks : ratio %u tasks per chain) !\n", nchains, totaltasks, totaltasks/nchains);
	free(is_entry_tab);
}

void init_problem(void)
{
	/* create the sparse input matrix */
	create_data();

	/* create a new codelet that will perform a SpMV on it */
	call_filters();
}

void print_results(void)
{
	unsigned row;

	for (row = 0; row < STARPU_MIN(size, 16); row++)
	{
		printf("%2.2f\t%2.2f\n", vector_in_ptr[row], vector_out_ptr[row]);
	}
}

int main(int argc, char *argv[])
{
	int ret;

	if (argc < 2)
	{
		FPRINTF(stderr, "usage : %s filename [tile size]\n", argv[0]);
		exit(-1);
	}

	if (argc == 3)
	{
		/* third argument is the tile size */
		char *argptr;
		r = strtol(argv[2], &argptr, 10);
		c = r;
	}

	inputfile = argv[1];

	/* start the runtime */
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	starpu_cublas_init();

	sem_init(&sem, 0, 0U);

	init_problem();

	launch_spmv_codelets();

	sem_wait(&sem);
	sem_destroy(&sem);

	unregister_data();
	print_results();

	double totalflop = 2.0*c*r*totaltasks;

	double timing = end - start;
	FPRINTF(stderr, "Computation took (in ms)\n");
	FPRINTF(stdout, "%2.2f\n", timing/1000);
	FPRINTF(stderr, "Flop %e\n", totalflop);
	FPRINTF(stderr, "GFlops : %2.2f\n", totalflop/timing/1000);

	return 0;
}
