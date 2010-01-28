/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include "pxlu.h"
#include "pxlu_kernels.h"

#define MPI_TAG11(k)	((1U << 16) | (k))
#define MPI_TAG12(k, j)	((2U << 16) | (j)*nblocks | (k))
#define MPI_TAG21(k, i)	((3U << 16) | (i)*nblocks | (k))

#define TAG11(k)	((starpu_tag_t)( (1ULL<<50) | (unsigned long long)(k)))
#define TAG12(k,i)	((starpu_tag_t)(((2ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21(k,j)	((starpu_tag_t)(((3ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j)	((starpu_tag_t)(((4ULL<<50) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))
#define TAG11_SAVE(k)	((starpu_tag_t)( (5ULL<<50) | (unsigned long long)(k)))
#define TAG12_SAVE(k,i)	((starpu_tag_t)(((6ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21_SAVE(k,j)	((starpu_tag_t)(((7ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))

#define TAG11_SAVE_PARTIAL(k)	((starpu_tag_t)( (8ULL<<50) | (unsigned long long)(k)))
#define TAG12_SAVE_PARTIAL(k,i)	((starpu_tag_t)(((9ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21_SAVE_PARTIAL(k,j)	((starpu_tag_t)(((10ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))

#define STARPU_TAG_INIT	((starpu_tag_t)(11ULL<<50))


static unsigned no_prio = 0;

static unsigned nblocks = 0;
static int rank = -1;
static int world_size = -1;

struct callback_arg {
	unsigned i, j, k;
};

/*
 *	Various
 */

static struct starpu_task *create_task(starpu_tag_t id)
{
	struct starpu_task *task = starpu_task_create();
		task->cl_arg = NULL;

	task->use_tag = 1;
	task->tag_id = id;

	return task;
}

/* Send handle to every node appearing in the mask, and unlock tag once the
 * transfers are done. */
static void send_data_to_mask(starpu_data_handle handle, int *rank_mask, int mpi_tag, starpu_tag_t tag)
{
	unsigned cnt = 0;

	int rank_array[world_size];
	int comm_array[world_size];
	int mpi_tag_array[world_size];
	starpu_data_handle handle_array[world_size];

	unsigned r;
	for (r = 0; r < world_size; r++)
	{
		if (rank_mask[r])
			rank_array[cnt++] = r;

		comm_array[r] = MPI_COMM_WORLD;
		mpi_tag_array[r] = mpi_tag;
		handle_array[r] = handle;
	}

	if (cnt == 0)
	{
		/* In case there is no message to send, we release the tag at
		 * once */
		starpu_tag_notify_from_apps(tag);
	}
	else {
		starpu_mpi_isend_array_detached_unlock_tag(cnt, handle_array,
				rank_array, mpi_tag_array, comm_array, tag);
	}
}

/* Initiate a receive request once all dependencies are fulfilled and unlock
 * tag 'unlocked_tag' once it's done. */

struct recv_when_done_callback_arg {
	int source;
	int mpi_tag;
	starpu_data_handle handle;
	starpu_tag_t unlocked_tag;
};

static void callback_receive_when_done(void *_arg)
{
	struct recv_when_done_callback_arg *arg = _arg;

	starpu_mpi_irecv_detached_unlock_tag(arg->handle, arg->source,
			arg->mpi_tag, MPI_COMM_WORLD, arg->unlocked_tag);

	free(arg);
}

static void receive_when_deps_are_done(unsigned ndeps, starpu_tag_t *deps_tags,
				int source, int mpi_tag,
				starpu_data_handle handle,
				starpu_tag_t partial_tag,
				starpu_tag_t unlocked_tag)
{
	struct recv_when_done_callback_arg *arg =
		malloc(sizeof(struct recv_when_done_callback_arg));
	
	arg->source = source;
	arg->mpi_tag = mpi_tag;
	arg->handle = handle;
	arg->unlocked_tag = unlocked_tag;

	if (ndeps == 0)
	{
		callback_receive_when_done(arg);
		return;
	}

	starpu_create_sync_task(partial_tag, ndeps, deps_tags,
					callback_receive_when_done, arg);
}

/*
 *	Task 11 (diagonal factorization)
 */

static void create_task_11_recv(unsigned k)
{
	unsigned i, j;

	/* The current node is not computing that task, so we receive the block
	 * with MPI */

	/* We don't issue a MPI receive request until everyone using the
	 * temporary buffer is done : 11_(k-1) can be used by 12_(k-1)j and
	 * 21(k-1)i with i,j >= k */
	unsigned ndeps = 0;
	starpu_tag_t tag_array[2*nblocks];
	
	if (k > 0)
	for (i = k; i < nblocks; i++)
	{
		if (rank == get_block_rank(i, k))
			tag_array[ndeps++] = TAG21(k-1, i);
	}

	if (k > 0)
	for (j = k; j < nblocks; j++)
	{
		if (rank == get_block_rank(k, j))
			tag_array[ndeps++] = TAG12(k-1, j);
	}
	
	
	int source = get_block_rank(k, j);
	starpu_data_handle block_handle = STARPU_PLU(get_tmp_11_block_handle)();
	int mpi_tag = MPI_TAG11(k);
	starpu_tag_t partial_tag = TAG11_SAVE_PARTIAL(k);
	starpu_tag_t unlocked_tag = TAG11_SAVE(k);

	fprintf(stderr, "NODE %d - 11 (%d) - recv when done ndeps %d - tag array %lx\n", rank, k, ndeps, tag_array[0]);
	receive_when_deps_are_done(ndeps, tag_array, source, mpi_tag, block_handle, partial_tag, unlocked_tag);
}

static void find_nodes_using_11(unsigned k, int *rank_mask)
{
	memset(rank_mask, 0, world_size*sizeof(int));

	/* Block 11_k is used to compute 12_kj + 12ki with i,j > k */
	unsigned i;
	for (i = k; i < nblocks; i++)
	{
		int r = get_block_rank(i, k);
		rank_mask[r] = 1;
	}

	unsigned j;
	for (j = k; j < nblocks; j++)
	{
		int r = get_block_rank(k, j);
		rank_mask[r] = 1;
	}
}

static void callback_task_11_real(void *_arg)
{
	struct callback_arg *arg = _arg;

	unsigned k = arg->k;

	/* Find all the nodes potentially requiring this block */
	int rank_mask[world_size];
	find_nodes_using_11(k, rank_mask);
	rank_mask[rank] = 0;

	/* Send the block to those nodes */
	starpu_data_handle block_handle = STARPU_PLU(get_block_handle)(k, k);
	starpu_tag_t tag = TAG11_SAVE(k);
	int mpi_tag = MPI_TAG11(k);
	send_data_to_mask(block_handle, rank_mask, mpi_tag, tag);
	
	free(arg);
}

static void create_task_11_real(unsigned k)
{
	struct starpu_task *task = create_task(TAG11(k));

	task->cl = &STARPU_PLU(cl11);

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = STARPU_PLU(get_block_handle)(k, k);
	task->buffers[0].mode = STARPU_RW;

	struct callback_arg *arg = malloc(sizeof(struct callback_arg));
		arg->k = k;

	task->callback_func = callback_task_11_real;
	task->callback_arg = arg;

	/* this is an important task */
	if (!no_prio)
		task->priority = MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}
	else {
		starpu_tag_declare_deps(TAG11(k), 1, STARPU_TAG_INIT);
	}

	starpu_submit_task(task);
}

static void create_task_11(unsigned k)
{
	if (get_block_rank(k, k) == rank)
	{
		fprintf(stderr, "CREATE real task 11(%d) on node %d\n", k, rank);
		create_task_11_real(k);
	}
	else {
		/* We don't handle the task, but perhaps we have to generate MPI transfers. */
		int rank_mask[world_size];
		find_nodes_using_11(k, rank_mask);
		
		if (rank_mask[rank])
		{
			fprintf(stderr, "create RECV task 11(%d) on node %d\n", k, rank);
			create_task_11_recv(k);
		}
		else {
			fprintf(stderr, "Node %d needs not 11(%d)\n", rank, k);
		}
	}
}



/*
 *	Task 12 (Update lower left (TRSM))
 */

static void create_task_12_recv(unsigned k, unsigned j)
{
	unsigned i;

	/* The current node is not computing that task, so we receive the block
	 * with MPI */

	/* We don't issue a MPI receive request until everyone using the
	 * temporary buffer is done : 12_(k-1)j can be used by 22_(k-1)ij with
	 * i >= k */
	unsigned ndeps = 0;
	starpu_tag_t tag_array[nblocks];
	
	if (k > 0)
	for (i = k; i < nblocks; i++)
	{
		if (rank == get_block_rank(i, j))
			tag_array[ndeps++] = TAG22(k-1, i, j);
	}
	
	int source = get_block_rank(k, j);
	starpu_data_handle block_handle = STARPU_PLU(get_tmp_12_block_handle)(j);
	int mpi_tag = MPI_TAG12(k, j);
	starpu_tag_t partial_tag = TAG12_SAVE_PARTIAL(k, j);
	starpu_tag_t unlocked_tag = TAG12_SAVE(k, j);

	receive_when_deps_are_done(ndeps, tag_array, source, mpi_tag, block_handle, partial_tag, unlocked_tag);
}

static void find_nodes_using_12(unsigned k, unsigned j, int *rank_mask)
{
	memset(rank_mask, 0, world_size*sizeof(int));

	/* Block 12_kj is used to compute 22_kij with i > k */
	unsigned i;
	for (i = k; i < nblocks; i++)
	{
		int r = get_block_rank(i, j);
		rank_mask[r] = 1;
	}
}

static void callback_task_12_real(void *_arg)
{
	struct callback_arg *arg = _arg;

	unsigned k = arg->k;
	unsigned j = arg->j;

	/* Find all the nodes potentially requiring this block */
	int rank_mask[world_size];
	find_nodes_using_12(k, j, rank_mask);
	rank_mask[rank] = 0;

	/* Send the block to those nodes */
	starpu_data_handle block_handle = STARPU_PLU(get_block_handle)(j, k);
	starpu_tag_t tag = TAG12_SAVE(k, j);
	int mpi_tag = MPI_TAG12(k, j);
	send_data_to_mask(block_handle, rank_mask, mpi_tag, tag);
	
	free(arg);
}

static void create_task_12_real(unsigned k, unsigned j)
{
	struct starpu_task *task = create_task(TAG12(k, j));
	
	task->cl = &STARPU_PLU(cl12);

	/* which sub-data is manipulated ? */
	starpu_data_handle diag_block;
	if (get_block_rank(k, k) == rank)
		diag_block = STARPU_PLU(get_block_handle)(k, k);
	else 
		diag_block = STARPU_PLU(get_tmp_11_block_handle)();

	task->buffers[0].handle = diag_block; 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = STARPU_PLU(get_block_handle)(j, k); 
	task->buffers[1].mode = STARPU_RW;

	struct callback_arg *arg = malloc(sizeof(struct callback_arg));
		arg->j = j;
		arg->k = k;

	task->callback_func = callback_task_12_real;
	task->callback_arg = arg;

	if (!no_prio && (j == k+1)) {
		task->priority = MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG12(k, j), 2, TAG11_SAVE(k), TAG22(k-1, k, j));
	}
	else {
		starpu_tag_declare_deps(TAG12(k, j), 1, TAG11_SAVE(k));
	}

	starpu_submit_task(task);
}

static void create_task_12(unsigned k, unsigned j)
{
	if (get_block_rank(k, j) == rank)
	{
		fprintf(stderr, "CREATE real task 12(k = %d, j = %d) on node %d\n", k, j, rank);
		create_task_12_real(k, j);
	}
	else {
		/* We don't handle the task, but perhaps we have to generate MPI transfers. */
		int rank_mask[world_size];
		find_nodes_using_12(k, j, rank_mask);
		
		if (rank_mask[rank])
		{
			fprintf(stderr, "create RECV task 12(k = %d, j = %d) on node %d\n", k, j, rank);
			create_task_12_recv(k, j);
		}
		else {
			fprintf(stderr, "Node %d needs not 12(k=%d, i=%d)\n", rank, k, j);
		}
	}
}

/*
 *	Task 21 (Update upper right (TRSM))
 */

static void create_task_21_recv(unsigned k, unsigned i)
{
	unsigned j;

	/* The current node is not computing that task, so we receive the block
	 * with MPI */

	/* We don't issue a MPI receive request until everyone using the
	 * temporary buffer is done : 21_(k-1)i can be used by 22_(k-1)ij with
	 * j >= k */
	unsigned ndeps = 0;
	starpu_tag_t tag_array[nblocks];
	
	if (k > 0)
	for (j = k; j < nblocks; j++)
	{
		if (rank == get_block_rank(i, j))
			tag_array[ndeps++] = TAG22(k-1, i, j);
	}
	
	int source = get_block_rank(i, k);
	starpu_data_handle block_handle = STARPU_PLU(get_tmp_21_block_handle)(i);
	int mpi_tag = MPI_TAG21(k, i);
	starpu_tag_t partial_tag = TAG21_SAVE_PARTIAL(k, i);
	starpu_tag_t unlocked_tag = TAG21_SAVE(k, i);

	fprintf(stderr, "NODE %d - 21 (%d, %d) - recv when done ndeps %d - tag array %lx\n", rank, k, i, ndeps, tag_array[0]);
	receive_when_deps_are_done(ndeps, tag_array, source, mpi_tag, block_handle, partial_tag, unlocked_tag);
}

static void find_nodes_using_21(unsigned k, unsigned i, int *rank_mask)
{
	memset(rank_mask, 0, world_size*sizeof(int));

	/* Block 21_ki is used to compute 22_kij with j > k */
	unsigned j;
	for (j = k; j < nblocks; j++)
	{
		int r = get_block_rank(i, j);
		rank_mask[r] = 1;
	}
}

static void callback_task_21_real(void *_arg)
{
	struct callback_arg *arg = _arg;

	unsigned k = arg->k;
	unsigned i = arg->i;

	/* Find all the nodes potentially requiring this block */
	int rank_mask[world_size];
	find_nodes_using_21(k, i, rank_mask);
	rank_mask[rank] = 0;

	/* Send the block to those nodes */
	starpu_data_handle block_handle = STARPU_PLU(get_block_handle)(k, i);
	starpu_tag_t tag = TAG21_SAVE(k, i);
	int mpi_tag = MPI_TAG21(k, i);
	send_data_to_mask(block_handle, rank_mask, mpi_tag, tag);
	
	free(arg);
}

static void create_task_21_real(unsigned k, unsigned i)
{
	struct starpu_task *task = create_task(TAG21(k, i));

	task->cl = &STARPU_PLU(cl21);
	
	/* which sub-data is manipulated ? */
	starpu_data_handle diag_block;
	if (get_block_rank(k, k) == rank)
		diag_block = STARPU_PLU(get_block_handle)(k, k);
	else 
		diag_block = STARPU_PLU(get_tmp_11_block_handle)();

	task->buffers[0].handle = diag_block; 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = STARPU_PLU(get_block_handle)(k, i);
	task->buffers[1].mode = STARPU_RW;

	struct callback_arg *arg = malloc(sizeof(struct callback_arg));
		arg->i = i;
		arg->k = k;

	task->callback_func = callback_task_21_real;
	task->callback_arg = arg;

	if (!no_prio && (i == k+1)) {
		task->priority = MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG21(k, i), 2, TAG11_SAVE(k), TAG22(k-1, i, k));
	}
	else {
		starpu_tag_declare_deps(TAG21(k, i), 1, TAG11_SAVE(k));
	}

	starpu_submit_task(task);
}

static void create_task_21(unsigned k, unsigned i)
{
	if (get_block_rank(i, k) == rank)
	{
		fprintf(stderr, "CREATE real task 21(k = %d, i = %d) on node %d\n", k, i, rank);
		create_task_21_real(k, i);
	}
	else {
		/* We don't handle the task, but perhaps we have to generate MPI transfers. */
		int rank_mask[world_size];
		find_nodes_using_21(k, i, rank_mask);
		
		if (rank_mask[rank])
		{
			fprintf(stderr, "create RECV task 21(k = %d, i = %d) on node %d\n", k, i, rank);
			create_task_21_recv(k, i);
		}
		else {
			fprintf(stderr, "Node %d needs not 21(k=%d, i=%d)\n", rank, k,i);
		}
	}
}

/*
 *	Task 22 (GEMM)
 */

static void create_task_22_real(unsigned k, unsigned i, unsigned j)
{
//	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j));

	struct starpu_task *task = create_task(TAG22(k, i, j));

	task->cl = &STARPU_PLU(cl22);

	/* which sub-data is manipulated ? */

	/* produced by TAG21_SAVE(k, i) */ 
	starpu_data_handle block21;
	if (get_block_rank(i, k) == rank)
		block21 = STARPU_PLU(get_block_handle)(k, i);
	else 
		block21 = STARPU_PLU(get_tmp_21_block_handle)(i);

	task->buffers[0].handle = block21;
	task->buffers[0].mode = STARPU_R;

	/* produced by TAG12_SAVE(k, j) */
	starpu_data_handle block12;
	if (get_block_rank(k, j) == rank)
		block12 = STARPU_PLU(get_block_handle)(j, k);
	else 
		block12 = STARPU_PLU(get_tmp_12_block_handle)(j);

	task->buffers[1].handle = block12;
	task->buffers[1].mode = STARPU_R;

	/* produced by TAG22(k-1, i, j) */
	task->buffers[2].handle = STARPU_PLU(get_block_handle)(j, i);
	task->buffers[2].mode = STARPU_RW;

	if (!no_prio &&  (i == k + 1) && (j == k +1) ) {
		task->priority = MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG12_SAVE(k, j), TAG21_SAVE(k, i));
	}
	else {
		starpu_tag_declare_deps(TAG22(k, i, j), 2, TAG12_SAVE(k, j), TAG21_SAVE(k, i));
	}

	starpu_submit_task(task);
}

static void create_task_22(unsigned k, unsigned i, unsigned j)
{
	if (get_block_rank(i, j) == rank)
	{
		fprintf(stderr, "CREATE real task 22(k = %d, i = %d, j = %d) on node %d\n", k, i, j, rank);
		create_task_22_real(k, i, j);
	}
	else {
		fprintf(stderr, "Node %d needs not 22(k=%d, i=%d, j = %d)\n", rank, k,i,j);
	}
}



/*
 *	code to bootstrap the factorization 
 */

void STARPU_PLU(plu_main)(unsigned _nblocks, int _rank, int _world_size)
{
	struct timeval start;
	struct timeval end;

	nblocks = _nblocks;
	rank = _rank;
	world_size = _world_size;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		create_task_11(k);

		for (i = k+1; i<nblocks; i++)
		{
			create_task_12(k, i);
			create_task_21(k, i);
		}

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
				create_task_22(k, i, j);
			}
		}
	}

	/* schedule the codelet */
	gettimeofday(&start, NULL);

	fprintf(stderr, "Rank %d GO\n", rank);
	starpu_tag_notify_from_apps(STARPU_TAG_INIT);

	/* stall the application until the end of computations : note that it
	 * may be liberated explicitely by MPI */
	starpu_tag_wait(TAG11(nblocks-1));

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

//	unsigned n = size;
//	double flop = (2.0f*n*n*n)/3.0f;
//	fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}
