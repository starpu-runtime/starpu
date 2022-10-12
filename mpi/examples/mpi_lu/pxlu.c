/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "pxlu.h"
#include "pxlu_kernels.h"
#include <sys/time.h>

#define MPI_TAG_GETRF(k)	((1U << 16) | (k))
#define MPI_TAG_TRSM_LL(k, j)	((2U << 16) | (k)<<8 | (j))
#define MPI_TAG_TRSM_RU(k, i)	((3U << 16) | (i)<<8 | (k))

// GETRF   TRSM_RU
// TRSM_LL GEMM

#define TAG_GETRF(k)	((starpu_tag_t)((1ULL<<50) | (unsigned long long)(k)))
#define TAG_TRSM_LL(k,j)	((starpu_tag_t)(((2ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_TRSM_RU(k,i)	((starpu_tag_t)(((3ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG_GEMM(k,i,j)	((starpu_tag_t)(((4ULL<<50) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))
#define TAG_GETRF_SAVE(k)	((starpu_tag_t)((5ULL<<50) | (unsigned long long)(k)))
#define TAG_TRSM_LL_SAVE(k,j)	((starpu_tag_t)(((6ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_TRSM_RU_SAVE(k,i)	((starpu_tag_t)(((7ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))

#define TAG_GETRF_SAVE_PARTIAL(k)	((starpu_tag_t)((8ULL<<50) | (unsigned long long)(k)))
#define TAG_TRSM_LL_SAVE_PARTIAL(k,j)	((starpu_tag_t)(((9ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_TRSM_RU_SAVE_PARTIAL(k,i)	((starpu_tag_t)(((10ULL<<50) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))

#define STARPU_TAG_INIT	((starpu_tag_t)(11ULL<<50))

//#define VERBOSE_INIT	1

//#define DEBUG	1

static unsigned no_prio = 0;

static unsigned nblocks = 0;
static int rank = -1;
static int world_size = -1;

struct callback_arg
{
	unsigned i, j, k;
};

/*
 *	Various
 */

static struct debug_info *create_debug_info(unsigned i, unsigned j, unsigned k)
{
	struct debug_info *info = malloc(sizeof(struct debug_info));

	info->i = i;
	info->j = j;
	info->k = k;

	return info;
}

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
static void send_data_to_mask(starpu_data_handle_t handle, int *rank_mask, starpu_mpi_tag_t mpi_tag, starpu_tag_t tag)
{
	unsigned cnt = 0;

	STARPU_ASSERT(handle != STARPU_POISON_PTR);

	int rank_array[world_size];
	MPI_Comm comm_array[world_size];
	starpu_mpi_tag_t mpi_tag_array[world_size];
	starpu_data_handle_t handle_array[world_size];

	int r;
	for (r = 0; r < world_size; r++)
	{
		if (rank_mask[r])
		{
			rank_array[cnt] = r;

			comm_array[cnt] = MPI_COMM_WORLD;
			mpi_tag_array[cnt] = mpi_tag;
			handle_array[cnt] = handle;
			cnt++;
		}
	}

	if (cnt == 0)
	{
		/* In case there is no message to send, we release the tag at
		 * once */
		starpu_tag_notify_from_apps(tag);
	}
	else
	{
		int ret = starpu_mpi_isend_array_detached_unlock_tag(cnt, handle_array,
								     rank_array, mpi_tag_array, comm_array, tag);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_array_detached_unlock_tag");
	}
}

/* Initiate a receive request once all dependencies are fulfilled and unlock
 * tag 'unlocked_tag' once it's done. */

struct recv_when_done_callback_arg
{
	int source;
	starpu_mpi_tag_t mpi_tag;
	starpu_data_handle_t handle;
	starpu_tag_t unlocked_tag;
};

static void callback_receive_when_done(void *_arg)
{
	struct recv_when_done_callback_arg *arg = _arg;

	int ret = starpu_mpi_irecv_detached_unlock_tag(arg->handle, arg->source,
						       arg->mpi_tag, MPI_COMM_WORLD, arg->unlocked_tag);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv_detached_unlock_tag");

	free(arg);
}

static void receive_when_deps_are_done(unsigned ndeps, starpu_tag_t *deps_tags,
				       int source, starpu_mpi_tag_t mpi_tag,
				       starpu_data_handle_t handle,
				       starpu_tag_t partial_tag,
				       starpu_tag_t unlocked_tag)
{
	STARPU_ASSERT(handle != STARPU_POISON_PTR);

	struct recv_when_done_callback_arg *arg = malloc(sizeof(struct recv_when_done_callback_arg));

	arg->source = source;
	arg->mpi_tag = mpi_tag;
	arg->handle = handle;
	arg->unlocked_tag = unlocked_tag;

	if (ndeps == 0)
	{
		callback_receive_when_done(arg);
		return;
	}

	starpu_create_sync_task(partial_tag, ndeps, deps_tags, callback_receive_when_done, arg);
}

/*
 *	Task GETRF (diagonal factorization)
 */

static void create_task_getrf_recv(unsigned k)
{
	/* The current node is not computing that task, so we receive the block
	 * with MPI */

	/* We don't issue a MPI receive request until everyone using the
	 * temporary buffer is done : 11_(k-1) can be used by 12_(k-1)j and
	 * 21(k-1)i with i,j >= k */
	unsigned ndeps = 0;
	starpu_tag_t tag_array[2*nblocks];

#ifdef SINGLE_TMP11
	if (k > 0)
	{
		unsigned i;
		for (i = (k-1)+1; i < nblocks; i++)
		{
			if (rank == get_block_rank(i, k-1))
				tag_array[ndeps++] = TAG_TRSM_RU(k-1, i);
		}

		unsigned j;
		for (j = (k-1)+1; j < nblocks; j++)
		{
			if (rank == get_block_rank(k-1, j))
				tag_array[ndeps++] = TAG_TRSM_LL(k-1, j);
		}
	}
#endif

	int source = get_block_rank(k, k);
#ifdef SINGLE_TMP11
	starpu_data_handle_t block_handle = STARPU_PLU(get_tmp_11_block_handle)();
#else
	starpu_data_handle_t block_handle = STARPU_PLU(get_tmp_11_block_handle)(k);
#endif
	starpu_mpi_tag_t mpi_tag = MPI_TAG_GETRF(k);
	starpu_tag_t partial_tag = TAG_GETRF_SAVE_PARTIAL(k);
	starpu_tag_t unlocked_tag = TAG_GETRF_SAVE(k);

//	fprintf(stderr, "NODE %d - 11 (%d) - recv when done ndeps %d - tag array %lx\n", rank, k, ndeps, tag_array[0]);
	receive_when_deps_are_done(ndeps, tag_array, source, mpi_tag, block_handle, partial_tag, unlocked_tag);
}

static void find_nodes_using_11(unsigned k, int *rank_mask)
{
	memset(rank_mask, 0, world_size*sizeof(int));

	/* Block 11_k is used to compute 12_kj + 12ki with i,j > k */
	unsigned i;
	for (i = k+1; i < nblocks; i++)
	{
		int r = get_block_rank(i, k);
		rank_mask[r] = 1;
	}

	unsigned j;
	for (j = k+1; j < nblocks; j++)
	{
		int r = get_block_rank(k, j);
		rank_mask[r] = 1;
	}
}

static void callback_task_getrf_real(void *_arg)
{
	struct callback_arg *arg = _arg;

	unsigned k = arg->k;

	/* Find all the nodes potentially requiring this block */
	int rank_mask[world_size];
	find_nodes_using_11(k, rank_mask);
	rank_mask[rank] = 0;

	/* Send the block to those nodes */
	starpu_data_handle_t block_handle = STARPU_PLU(get_block_handle)(k, k);
	starpu_tag_t tag = TAG_GETRF_SAVE(k);
	starpu_mpi_tag_t mpi_tag = MPI_TAG_GETRF(k);
	send_data_to_mask(block_handle, rank_mask, mpi_tag, tag);

	free(arg);
}

static void create_task_getrf_real(unsigned k)
{
	struct starpu_task *task = create_task(TAG_GETRF(k));

	task->cl = &STARPU_PLU(cl_getrf);
	task->color = 0xffff00;

	task->cl_arg = create_debug_info(k, k, k);
	task->cl_arg_free = 1;

	/* which sub-data is manipulated ? */
	task->handles[0] = STARPU_PLU(get_block_handle)(k, k);

	struct callback_arg *arg = malloc(sizeof(struct callback_arg));
		arg->k = k;

	task->callback_func = callback_task_getrf_real;
	task->callback_arg = arg;

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_GETRF(k), 1, TAG_GEMM(k-1, k, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG_GETRF(k), 1, STARPU_TAG_INIT);
	}

	int ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static void create_task_getrf(unsigned k)
{
	if (get_block_rank(k, k) == rank)
	{
#ifdef VERBOSE_INIT
		fprintf(stderr, "CREATE real task 11(%u) (TAG_GETRF_SAVE(%u) = %llux) on node %d\n", k, k, (unsigned long long) TAG_GETRF_SAVE(k), rank);
#endif
		create_task_getrf_real(k);
	}
	else
	{
		/* We don't handle the task, but perhaps we have to generate MPI transfers. */
		int rank_mask[world_size];
		find_nodes_using_11(k, rank_mask);

		if (rank_mask[rank])
		{
#ifdef VERBOSE_INIT
			fprintf(stderr, "create RECV task 11(%u) on node %d\n", k, rank);
#endif
			create_task_getrf_recv(k);
		}
		else
		{
#ifdef VERBOSE_INIT
			fprintf(stderr, "Node %d needs not 11(%u)\n", rank, k);
#endif
		}
	}
}



/*
 *	Task TRSM_LL
 */

static void create_task_trsm_ll_recv(unsigned k, unsigned j)
{
	/* The current node is not computing that task, so we receive the block
	 * with MPI */

	/* We don't issue a MPI receive request until everyone using the
	 * temporary buffer is done : 12_(k-1)j can be used by 22_(k-1)ij with
	 * i >= k */
	unsigned ndeps = 0;
	starpu_tag_t tag_array[nblocks];

	unsigned start;
	unsigned bound;

#ifdef SINGLE_TMP1221
	bound = 0;
	start = (k-1)+1;
#else
	bound = 1;
	start = (k-2)+1;
#endif

	if (k > bound)
	{
		unsigned i;
		for (i = start; i < nblocks; i++)
		{
			if (rank == get_block_rank(i, j))
#ifdef SINGLE_TMP1221
				tag_array[ndeps++] = TAG_GEMM(k-1, i, j);
#else
				tag_array[ndeps++] = TAG_GEMM(k-2, i, j);
#endif
		}
	}

	int source = get_block_rank(k, j);
#ifdef SINGLE_TMP1221
	starpu_data_handle_t block_handle = STARPU_PLU(get_tmp_12_block_handle)(j);
#else
	starpu_data_handle_t block_handle = STARPU_PLU(get_tmp_12_block_handle)(j,k);
#endif
	starpu_mpi_tag_t mpi_tag = MPI_TAG_TRSM_LL(k, j);
	starpu_tag_t partial_tag = TAG_TRSM_LL_SAVE_PARTIAL(k, j);
	starpu_tag_t unlocked_tag = TAG_TRSM_LL_SAVE(k, j);

	receive_when_deps_are_done(ndeps, tag_array, source, mpi_tag, block_handle, partial_tag, unlocked_tag);
}

static void find_nodes_using_12(unsigned k, unsigned j, int *rank_mask)
{
	memset(rank_mask, 0, world_size*sizeof(int));

	/* Block 12_kj is used to compute 22_kij with i > k */
	unsigned i;
	for (i = k+1; i < nblocks; i++)
	{
		int r = get_block_rank(i, j);
		rank_mask[r] = 1;
	}
}

static void callback_task_trsm_ll_real(void *_arg)
{
	struct callback_arg *arg = _arg;

	unsigned k = arg->k;
	unsigned j = arg->j;

	/* Find all the nodes potentially requiring this block */
	int rank_mask[world_size];
	find_nodes_using_12(k, j, rank_mask);
	rank_mask[rank] = 0;

	/* Send the block to those nodes */
	starpu_data_handle_t block_handle = STARPU_PLU(get_block_handle)(k, j);
	starpu_tag_t tag = TAG_TRSM_LL_SAVE(k, j);
	starpu_mpi_tag_t mpi_tag = MPI_TAG_TRSM_LL(k, j);
	send_data_to_mask(block_handle, rank_mask, mpi_tag, tag);

	free(arg);
}

static void create_task_trsm_ll_real(unsigned k, unsigned j)
{
	struct starpu_task *task = create_task(TAG_TRSM_LL(k, j));

#ifdef STARPU_DEVEL
#warning temporary fix :/
#endif
//	task->cl = &STARPU_PLU(cl_trsm_ll);
	task->cl = &STARPU_PLU(cl_trsm_ru);
	task->color = 0x8080ff;

	task->cl_arg = create_debug_info(j, j, k);
	task->cl_arg_free = 1;

	unsigned diag_block_is_local = (get_block_rank(k, k) == rank);

	starpu_tag_t tag_11_dep;

	/* which sub-data is manipulated ? */
	starpu_data_handle_t diag_block;
	if (diag_block_is_local)
	{
		diag_block = STARPU_PLU(get_block_handle)(k, k);
		tag_11_dep = TAG_GETRF(k);
	}
	else
	{
#ifdef SINGLE_TMP11
		diag_block = STARPU_PLU(get_tmp_11_block_handle)();
#else
		diag_block = STARPU_PLU(get_tmp_11_block_handle)(k);
#endif
		tag_11_dep = TAG_GETRF_SAVE(k);
	}

	task->handles[0] = diag_block;
	task->handles[1] = STARPU_PLU(get_block_handle)(k, j);

	STARPU_ASSERT(get_block_rank(k, j) == rank);

	STARPU_ASSERT(task->handles[0] != STARPU_POISON_PTR);
	STARPU_ASSERT(task->handles[1] != STARPU_POISON_PTR);

	struct callback_arg *arg = malloc(sizeof(struct callback_arg));
		arg->j = j;
		arg->k = k;

	task->callback_func = callback_task_trsm_ll_real;
	task->callback_arg = arg;

	if (!no_prio && (j == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM_LL(k, j), 2, tag_11_dep, TAG_GEMM(k-1, k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM_LL(k, j), 1, tag_11_dep);
	}

	int ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static void create_task_trsm_ll(unsigned k, unsigned j)
{
	if (get_block_rank(k, j) == rank)
	{
#ifdef VERBOSE_INIT
		fprintf(stderr, "CREATE real task 12(k = %u, j = %u) on node %d\n", k, j, rank);
#endif
		create_task_trsm_ll_real(k, j);
	}
	else
	{
		/* We don't handle the task, but perhaps we have to generate MPI transfers. */
		int rank_mask[world_size];
		find_nodes_using_12(k, j, rank_mask);

		if (rank_mask[rank])
		{
#ifdef VERBOSE_INIT
			fprintf(stderr, "create RECV task 12(k = %u, j = %u) on node %d\n", k, j, rank);
#endif
			create_task_trsm_ll_recv(k, j);
		}
		else
		{
#ifdef VERBOSE_INIT
			fprintf(stderr, "Node %d needs not 12(k=%u, i=%u)\n", rank, k, j);
#endif
		}
	}
}

/*
 *	Task TRSM_RU
 */

static void create_task_trsm_ru_recv(unsigned k, unsigned i)
{
	/* The current node is not computing that task, so we receive the block
	 * with MPI */

	/* We don't issue a MPI receive request until everyone using the
	 * temporary buffer is done : 21_(k-1)i can be used by 22_(k-1)ij with
	 * j >= k */
	unsigned ndeps = 0;
	starpu_tag_t tag_array[nblocks];

	unsigned bound;
	unsigned start;

#ifdef SINGLE_TMP1221
	bound = 0;
	start = (k-1)+1;
#else
	bound = 1;
	start = (k-2)+1;
#endif
	if (k > bound)
	{
		unsigned j;
		for (j = start; j < nblocks; j++)
		{
			if (rank == get_block_rank(i, j))
#ifdef SINGLE_TMP1221
				tag_array[ndeps++] = TAG_GEMM(k-1, i, j);
#else
				tag_array[ndeps++] = TAG_GEMM(k-2, i, j);
#endif
		}
	}

	int source = get_block_rank(i, k);
#ifdef SINGLE_TMP1221
	starpu_data_handle_t block_handle = STARPU_PLU(get_tmp_21_block_handle)(i);
#else
	starpu_data_handle_t block_handle = STARPU_PLU(get_tmp_21_block_handle)(i, k);
#endif
	starpu_mpi_tag_t mpi_tag = MPI_TAG_TRSM_RU(k, i);
	starpu_tag_t partial_tag = TAG_TRSM_RU_SAVE_PARTIAL(k, i);
	starpu_tag_t unlocked_tag = TAG_TRSM_RU_SAVE(k, i);

//	fprintf(stderr, "NODE %d - 21 (%d, %d) - recv when done ndeps %d - tag array %lx\n", rank, k, i, ndeps, tag_array[0]);
	receive_when_deps_are_done(ndeps, tag_array, source, mpi_tag, block_handle, partial_tag, unlocked_tag);
}

static void find_nodes_using_21(unsigned k, unsigned i, int *rank_mask)
{
	memset(rank_mask, 0, world_size*sizeof(int));

	/* Block 21_ki is used to compute 22_kij with j > k */
	unsigned j;
	for (j = k+1; j < nblocks; j++)
	{
		int r = get_block_rank(i, j);
		rank_mask[r] = 1;
	}
}

static void callback_task_trsm_ru_real(void *_arg)
{
	struct callback_arg *arg = _arg;

	unsigned k = arg->k;
	unsigned i = arg->i;

	/* Find all the nodes potentially requiring this block */
	int rank_mask[world_size];
	find_nodes_using_21(k, i, rank_mask);
	rank_mask[rank] = 0;

	/* Send the block to those nodes */
	starpu_data_handle_t block_handle = STARPU_PLU(get_block_handle)(i, k);
	starpu_tag_t tag = TAG_TRSM_RU_SAVE(k, i);
	starpu_mpi_tag_t mpi_tag = MPI_TAG_TRSM_RU(k, i);
	send_data_to_mask(block_handle, rank_mask, mpi_tag, tag);

	free(arg);
}

static void create_task_trsm_ru_real(unsigned k, unsigned i)
{
	struct starpu_task *task = create_task(TAG_TRSM_RU(k, i));

#ifdef STARPU_DEVEL
#warning temporary fix
#endif
//	task->cl = &STARPU_PLU(cl_trsm_ru);
	task->cl = &STARPU_PLU(cl_trsm_ll);
	task->color = 0x8080c0;

	task->cl_arg = create_debug_info(i, i, k);
	task->cl_arg_free = 1;

	unsigned diag_block_is_local = (get_block_rank(k, k) == rank);

	starpu_tag_t tag_11_dep;

	/* which sub-data is manipulated ? */
	starpu_data_handle_t diag_block;
	if (diag_block_is_local)
	{
		diag_block = STARPU_PLU(get_block_handle)(k, k);
		tag_11_dep = TAG_GETRF(k);
	}
	else
	{
#ifdef SINGLE_TMP11
		diag_block = STARPU_PLU(get_tmp_11_block_handle)();
#else
		diag_block = STARPU_PLU(get_tmp_11_block_handle)(k);
#endif
		tag_11_dep = TAG_GETRF_SAVE(k);
	}

	task->handles[0] = diag_block;
	task->handles[1] = STARPU_PLU(get_block_handle)(i, k);

	STARPU_ASSERT(task->handles[0] != STARPU_POISON_PTR);
	STARPU_ASSERT(task->handles[1] != STARPU_POISON_PTR);

	struct callback_arg *arg = malloc(sizeof(struct callback_arg));
		arg->i = i;
		arg->k = k;

	task->callback_func = callback_task_trsm_ru_real;
	task->callback_arg = arg;

	if (!no_prio && (i == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM_RU(k, i), 2, tag_11_dep, TAG_GEMM(k-1, i, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM_RU(k, i), 1, tag_11_dep);
	}

	int ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static void create_task_trsm_ru(unsigned k, unsigned i)
{
	if (get_block_rank(i, k) == rank)
	{
#ifdef VERBOSE_INIT
		fprintf(stderr, "CREATE real task 21(k = %u, i = %u) on node %d\n", k, i, rank);
#endif
		create_task_trsm_ru_real(k, i);
	}
	else
	{
		/* We don't handle the task, but perhaps we have to generate MPI transfers. */
		int rank_mask[world_size];
		find_nodes_using_21(k, i, rank_mask);

		if (rank_mask[rank])
		{
#ifdef VERBOSE_INIT
			fprintf(stderr, "create RECV task 21(k = %u, i = %u) on node %d\n", k, i, rank);
#endif
			create_task_trsm_ru_recv(k, i);
		}
		else
		{
#ifdef VERBOSE_INIT
			fprintf(stderr, "Node %d needs not 21(k=%u, i=%u)\n", rank, k,i);
#endif
		}
	}
}

/*
 *	Task GEMM
 */

static void create_task_gemm_real(unsigned k, unsigned i, unsigned j)
{
//	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG_GEMM(k,i,j));

	struct starpu_task *task = create_task(TAG_GEMM(k, i, j));

	task->cl = &STARPU_PLU(cl_gemm);
	task->color = 0x00ff00;

	task->cl_arg = create_debug_info(i, j, k);
	task->cl_arg_free = 1;

	/* which sub-data is manipulated ? */

	/* produced by TAG_TRSM_RU_SAVE(k, i) */
	unsigned block21_is_local = (get_block_rank(i, k) == rank);
	starpu_tag_t tag_21_dep;

	starpu_data_handle_t block21;
	if (block21_is_local)
	{
		block21 = STARPU_PLU(get_block_handle)(i, k);
		tag_21_dep = TAG_TRSM_RU(k, i);
	}
	else
	{
#ifdef SINGLE_TMP1221
		block21 = STARPU_PLU(get_tmp_21_block_handle)(i);
#else
		block21 = STARPU_PLU(get_tmp_21_block_handle)(i, k);
#endif
		tag_21_dep = TAG_TRSM_RU_SAVE(k, i);
	}

	/* produced by TAG_TRSM_LL_SAVE(k, j) */
	unsigned block12_is_local = (get_block_rank(k, j) == rank);
	starpu_tag_t tag_12_dep;

	starpu_data_handle_t block12;
	if (block12_is_local)
	{
	//	block12 = STARPU_PLU(get_block_handle)(j, k);
		block12 = STARPU_PLU(get_block_handle)(k, j);
		tag_12_dep = TAG_TRSM_LL(k, j);
	}
	else
	{
#ifdef SINGLE_TMP1221
		block12 = STARPU_PLU(get_tmp_12_block_handle)(j);
#else
		block12 = STARPU_PLU(get_tmp_12_block_handle)(j, k);
#endif
		tag_12_dep = TAG_TRSM_LL_SAVE(k, j);
	}



#ifdef STARPU_DEVEL
#warning temporary fix :/
#endif
	//task->handles[0] = block21;
	task->handles[0] = block12;

	//task->handles[1] = block12;
	task->handles[1] = block21;

	/* produced by TAG_GEMM(k-1, i, j) */
	task->handles[2] = STARPU_PLU(get_block_handle)(i, j);

	STARPU_ASSERT(task->handles[0] != STARPU_POISON_PTR);
	STARPU_ASSERT(task->handles[1] != STARPU_POISON_PTR);
	STARPU_ASSERT(task->handles[2] != STARPU_POISON_PTR);

	if (!no_prio && (i == k + 1) && (j == k +1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 3, TAG_GEMM(k-1, i, j), tag_12_dep, tag_21_dep);
	}
	else
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 2, tag_12_dep, tag_21_dep);
	}

	int ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static void create_task_gemm(unsigned k, unsigned i, unsigned j)
{
	if (get_block_rank(i, j) == rank)
	{
	//	fprintf(stderr, "CREATE real task 22(k = %d, i = %d, j = %d) on node %d\n", k, i, j, rank);
		create_task_gemm_real(k, i, j);
	}
//	else
//	{
//		fprintf(stderr, "Node %d needs not 22(k=%d, i=%d, j = %d)\n", rank, k,i,j);
//	}
}

static void wait_tag_and_fetch_handle(starpu_tag_t tag, starpu_data_handle_t handle)
{
	STARPU_ASSERT(handle != STARPU_POISON_PTR);

	starpu_tag_wait(tag);
//	fprintf(stderr, "Rank %d : tag %lx is done\n", rank, tag);

	starpu_data_acquire(handle, STARPU_R);
	starpu_data_release(handle);

//	starpu_data_unregister(handle);
}

static void wait_termination(void)
{
	unsigned k, i, j;
	for (k = 0; k < nblocks; k++)
	{
		/* Wait task 11k if needed */
		if (get_block_rank(k, k) == rank)
		{
			starpu_data_handle_t diag_block = STARPU_PLU(get_block_handle)(k, k);
			wait_tag_and_fetch_handle(TAG_GETRF_SAVE(k), diag_block);
		}


		for (i = k + 1; i < nblocks; i++)
		{
			/* Wait task 21ki if needed */
			if (get_block_rank(i, k) == rank)
			{
				starpu_data_handle_t block21 = STARPU_PLU(get_block_handle)(i, k);
				//starpu_data_handle_t block21 = STARPU_PLU(get_block_handle)(k, i);
				//fprintf(stderr, "BLOCK21 i %d k %d -> handle %p\n", i, k, block21);
				wait_tag_and_fetch_handle(TAG_TRSM_RU_SAVE(k, i), block21);
			}
		}

		for (j = k + 1; j < nblocks; j++)
		{
			/* Wait task 12kj if needed */
			if (get_block_rank(k, j) == rank)
			{
				//starpu_data_handle_t block12 = STARPU_PLU(get_block_handle)(j, k);
				starpu_data_handle_t block12 = STARPU_PLU(get_block_handle)(k, j);
				//fprintf(stderr, "BLOCK12 j %d k %d -> handle %p\n", j, k, block12);
				wait_tag_and_fetch_handle(TAG_TRSM_LL_SAVE(k, j), block12);
			}
		}
	}
}

/*
 *	code to bootstrap the factorization
 */

double STARPU_PLU(plu_main)(unsigned _nblocks, int _rank, int _world_size, unsigned _no_prio)
{
	double start;
	double end;

	nblocks = _nblocks;
	rank = _rank;
	world_size = _world_size;
	no_prio = _no_prio;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		starpu_iteration_push(k);

		create_task_getrf(k);

		for (i = k+1; i<nblocks; i++)
		{
			create_task_trsm_ll(k, i);
			create_task_trsm_ru(k, i);
		}

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
				create_task_gemm(k, i, j);
			}
		}
		starpu_iteration_pop();
	}

	int barrier_ret = starpu_mpi_barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);

	/* schedule the codelet */
	start = starpu_timing_now();

	starpu_tag_notify_from_apps(STARPU_TAG_INIT);

	wait_termination();

	end = starpu_timing_now();

	double timing = end - start;

//	fprintf(stderr, "RANK %d -> took %f ms\n", rank, timing/1000);

	return timing;
}
