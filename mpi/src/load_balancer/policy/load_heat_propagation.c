/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi.h>
#include <mpi/starpu_mpi_tag.h>
#include <common/uthash.h>
#include <common/utils.h>
#include <math.h>
#include <starpu_mpi_private.h>
#include "load_balancer_policy.h"
#include "data_movements_interface.h"
#include "load_data_interface.h"
#include <common/config.h>

#if defined(STARPU_USE_MPI_MPI)

static starpu_mpi_tag_t TAG_LOAD(int n)
{
	return (n+1) << 24;
}

static starpu_mpi_tag_t TAG_MOV(int n)
{
	return (n+1) << 20;
}

/* Hash table of local pieces of data that has been moved out of the local MPI
 * node by the load balancer. All of these pieces of data must be migrated back
 * to the local node at the end of the execution. */
struct moved_data_entry
{
	UT_hash_handle hh;
	starpu_data_handle_t handle;
};

static struct moved_data_entry *mdh = NULL;

static starpu_pthread_mutex_t load_data_mutex;
static starpu_pthread_cond_t load_data_cond;

/* MPI infos */
static int my_rank;
static int world_size;

/* Number of neighbours of the local MPI node and their IDs. These are given by
 * the get_neighbors() method, and thus can be easily changed. */
static int *neighbor_ids = NULL;
static int nneighbors = 0;

/* Local load data */
static starpu_data_handle_t *load_data_handle = NULL;
static starpu_data_handle_t *load_data_handle_cpy = NULL;
/* Load data of neighbours */
static starpu_data_handle_t *neighbor_load_data_handles = NULL;

/* Table which contains a data_movements_handle for each MPI node of
 * MPI_COMM_WORLD. Since all the MPI nodes must be advised of any data
 * movement, this table will be used to perform communications of data
 * movements handles following an all-to-all model. */
static starpu_data_handle_t *data_movements_handles = NULL;

/* Load balancer interface which contains the application-specific methods for
 * the load balancer to use. */
static struct starpu_mpi_lb_conf *user_itf = NULL;

static double time_threshold = 20000;

/******************************************************************************
 *                              Balancing                                     *
 *****************************************************************************/


/* Decides which data has to move where, and fills the
 * data_movements_handles[my_rank] data handle from that.
 * In data :
 *  - local load_data_handle
 *  - nneighbors
 *  - neighbor_ids[nneighbors]
 *  - neighbor_load_data_handles[nneighbors]
 * Out data :
 *  - data_movements_handles[my_rank]
 */

static void balance(starpu_data_handle_t load_data_cpy)
{
	int less_loaded = -1;
	int n;
	double ref_elapsed_time;
	double my_elapsed_time = load_data_get_elapsed_time(load_data_cpy);

	/* Search for the less loaded neighbor */
	ref_elapsed_time = my_elapsed_time;
	for (n = 0; n < nneighbors; n++)
	{
		double elapsed_time = load_data_get_elapsed_time(neighbor_load_data_handles[n]);
		if (ref_elapsed_time > elapsed_time)
		{
			//fprintf(stderr,"Node%d: ref local time %lf vs neighbour%d time %lf\n", my_rank, ref_elapsed_time, neighbor_ids[n], elapsed_time);
			less_loaded = neighbor_ids[n];
			ref_elapsed_time = elapsed_time;
		}
	}

	/* We found it */
	if (less_loaded >= 0)
	{
		_STARPU_DEBUG("Less loaded found on node %d : %d\n", my_rank, less_loaded);
		double diff_time = my_elapsed_time - ref_elapsed_time;
		/* If the difference is higher than a time threshold, we move
		 * one data to the less loaded neighbour. */
		/* TODO: How to decide the time threshold ? */
		if ((time_threshold > 0) && (diff_time >= time_threshold))
		{
			starpu_data_handle_t *handles = NULL;
			int nhandles = 0;
			user_itf->get_data_unit_to_migrate(&handles, &nhandles, less_loaded);

			data_movements_reallocate_tables(data_movements_handles[my_rank], nhandles);

			if (nhandles)
			{
				starpu_mpi_tag_t *tags = data_movements_get_tags_table(data_movements_handles[my_rank]);
				int *ranks = data_movements_get_ranks_table(data_movements_handles[my_rank]);

				for (n = 0; n < nhandles; n++)
				{
					tags[n] = starpu_mpi_data_get_tag(handles[n]);
					ranks[n] = less_loaded;
				}

				free(handles);
			}
		}
		else
			data_movements_reallocate_tables(data_movements_handles[my_rank], 0);
	}
	else
		data_movements_reallocate_tables(data_movements_handles[my_rank], 0);
}

static void exchange_load_data_infos(starpu_data_handle_t load_data_cpy)
{
	int i;

	/* Allocate all requests and status for point-to-point communications */
	starpu_mpi_req load_send_req[nneighbors];
	starpu_mpi_req load_recv_req[nneighbors];

	MPI_Status load_send_status[nneighbors];
	MPI_Status load_recv_status[nneighbors];

	int flag;

	/* Send the local load data to neighbour nodes, and receive the remote load
	 * data from neighbour nodes */
	for (i = 0; i < nneighbors; i++)
	{
		//_STARPU_DEBUG("[node %d] sending and receiving with %i-th neighbor %i\n", my_rank, i, neighbor_ids[i]);
		starpu_mpi_isend(load_data_cpy, &load_send_req[i], neighbor_ids[i], TAG_LOAD(my_rank), MPI_COMM_WORLD);
		starpu_mpi_irecv(neighbor_load_data_handles[i], &load_recv_req[i], neighbor_ids[i], TAG_LOAD(neighbor_ids[i]), MPI_COMM_WORLD);
	}

	/* Wait for completion of all send requests */
	for (i = 0; i < nneighbors; i++)
	{
		flag = 0;
		while (!flag)
			starpu_mpi_test(&load_send_req[i], &flag, &load_send_status[i]);
	}

	/* Wait for completion of all receive requests */
	for (i = 0; i < nneighbors; i++)
	{
		flag = 0;
		while (!flag)
			starpu_mpi_test(&load_recv_req[i], &flag, &load_recv_status[i]);
	}
}

static void exchange_data_movements_infos()
{
	int i;

	/* Allocate all requests and status for point-to-point communications */
	starpu_mpi_req data_movements_send_req[world_size];
	starpu_mpi_req data_movements_recv_req[world_size];

	MPI_Status data_movements_send_status[world_size];
	MPI_Status data_movements_recv_status[world_size];

	int flag;

	/* Send the new ranks of local data to all other nodes, and receive the new
	 * ranks of all remote data from all other nodes */
	for (i = 0; i < world_size; i++)
	{
		if (i != my_rank)
		{
			//_STARPU_DEBUG("[node %d] Send and receive data movement with %d\n", my_rank, i);
			starpu_mpi_isend(data_movements_handles[my_rank], &data_movements_send_req[i], i, TAG_MOV(my_rank), MPI_COMM_WORLD);
			starpu_mpi_irecv(data_movements_handles[i], &data_movements_recv_req[i], i, TAG_MOV(i), MPI_COMM_WORLD);
		}
	}

	/* Wait for completion of all send requests */
	for (i = 0; i < world_size; i++)
	{
		if (i != my_rank)
		{
			//fprintf(stderr,"Wait for sending data movement of %d to %d\n", my_rank, i);
			flag = 0;
			while (!flag)
				starpu_mpi_test(&data_movements_send_req[i], &flag, &data_movements_send_status[i]);
		}
	}

	/* Wait for completion of all receive requests */
	for (i = 0; i < world_size; i++)
	{
		if (i != my_rank)
		{
			//fprintf(stderr,"Wait for recieving data movement from %d on %d\n", i, my_rank);
			flag = 0;
			while (!flag)
				starpu_mpi_test(&data_movements_recv_req[i], &flag, &data_movements_recv_status[i]);
		}
	}
}

static void update_data_ranks()
{
	int i,j;

	/* Update the new ranks for all concerned data */
	for (i = 0; i < world_size; i++)
	{
		int ndata_to_update = data_movements_get_size_tables(data_movements_handles[i]);
		if (ndata_to_update)
		{
			//fprintf(stderr,"Update %d data from table %d on node %d\n", ndata_to_update, i, my_rank);

			for (j = 0; j < ndata_to_update; j++)
			{
				starpu_data_handle_t handle = _starpu_mpi_tag_get_data_handle_from_tag((data_movements_get_tags_table(data_movements_handles[i]))[j]);
				STARPU_ASSERT(handle);
				int dst_rank = (data_movements_get_ranks_table(data_movements_handles[i]))[j];

				/* Save the fact that the data has been moved out of this node */
				if (i == my_rank)
				{
					struct moved_data_entry *md;
					_STARPU_MPI_MALLOC(md, sizeof(struct moved_data_entry));
					md->handle = handle;
					HASH_ADD_PTR(mdh, handle, md);
				}
				else if (dst_rank == my_rank)
				{
					/* The data has been moved out, and now is moved back, so
					 * update the state of the moved_data hash table to reflect
					 * this change */
					struct moved_data_entry *md = NULL;
					HASH_FIND_PTR(mdh, &handle, md);
					if (md)
					{
						HASH_DEL(mdh, md);
						free(md);
					}
				}

				//if (i == my_rank)
				//{
				//    if (dst_rank != my_rank)
				//        fprintf(stderr,"Move data %p (tag %d) from node %d to node %d\n", handle, (data_movements_get_tags_table(data_movements_handles[i]))[j], my_rank, dst_rank);
				//    else
				//        fprintf(stderr,"Bring back data %p (tag %d) from node %d on node %d\n", handle, (data_movements_get_tags_table(data_movements_handles[i]))[j], starpu_mpi_data_get_rank(handle), my_rank);
				//}

				_STARPU_DEBUG("Call of starpu_mpi_get_data_on_node(%"PRIi64",%d) on node %d\n", starpu_mpi_data_get_tag(handle), dst_rank, my_rank);

				/* Migrate the data handle */
				starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, handle, dst_rank, NULL, NULL);

				_STARPU_DEBUG("New rank (%d) of data %"PRIi64" upgraded on node %d\n", dst_rank, starpu_mpi_data_get_tag(handle), my_rank);
				starpu_mpi_data_set_rank_comm(handle, dst_rank, MPI_COMM_WORLD);
			}
		}
	}
}

static void clean_balance()
{
	int i;
	starpu_mpi_cache_flush(MPI_COMM_WORLD, *load_data_handle_cpy);
	for (i = 0; i < nneighbors; i++)
		starpu_mpi_cache_flush(MPI_COMM_WORLD, neighbor_load_data_handles[i]);
	for (i = 0; i < world_size; i++)
		starpu_mpi_cache_flush(MPI_COMM_WORLD, data_movements_handles[i]);
}

/* Core function of the load balancer. Computes from the load_data_cpy handle a
 * load balancing of the work to come (if needed), perform the necessary data
 * communications and negociate with the other nodes the rebalancing. */
static void heat_balance(starpu_data_handle_t load_data_cpy)
{
	/* Exchange load data handles with neighboring nodes */
	exchange_load_data_infos(load_data_cpy);

	/* Determine if this node should sent data to other nodes :
	 * which ones, how much data */
	balance(load_data_cpy);

	/* Exchange data movements with neighboring nodes */
	exchange_data_movements_infos();

	/* Perform data movements */
	update_data_ranks();

	/* Clean the data handles to properly launch the next balance phase */
	clean_balance();
}

/******************************************************************************
 *                      Heat Load Balancer Entry Points                       *
 *****************************************************************************/

static void submitted_task_heat(struct starpu_task *task)
{
	load_data_inc_nsubmitted_tasks(*load_data_handle);
	//if (load_data_get_nsubmitted_tasks(*load_data_handle) > task->tag_id)
	//{
	//    fprintf(stderr,"Error : nsubmitted_tasks (%d) > tag_id (%lld) ! \n", load_data_get_nsubmitted_tasks(*load_data_handle), (long long int)task->tag_id);
	//    STARPU_ASSERT(0);
	//}

	int phase = load_data_get_current_phase(*load_data_handle);
	/* Numbering of tasks in StarPU-MPI should be given by the application with
	 * the STARPU_TAG_ONLY insert task option for now. */
	/* TODO: Properly implement a solution for numbering tasks in StarPU-MPI */
	if (((int)task->tag_id / load_data_get_sleep_threshold(*load_data_handle)) > phase)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&load_data_mutex);
		load_data_update_wakeup_cond(*load_data_handle);
		//fprintf(stderr,"Node %d sleep on tag %lld\n", my_rank, (long long int)task->tag_id);
		//if (load_data_get_nsubmitted_tasks(*load_data_handle) < load_data_get_wakeup_threshold(*load_data_handle))
		//{
		//    fprintf(stderr,"Error : nsubmitted_tasks (%d) lower than wakeup_threshold (%d) !\n", load_data_get_nsubmitted_tasks(*load_data_handle), load_data_get_wakeup_threshold(*load_data_handle));
		//    STARPU_ASSERT(0);
		//}

		if (load_data_get_wakeup_threshold(*load_data_handle) > load_data_get_nfinished_tasks(*load_data_handle))
			STARPU_PTHREAD_COND_WAIT(&load_data_cond, &load_data_mutex);

		load_data_next_phase(*load_data_handle);

		/* Register a copy of the load data at this moment, to allow to compute
		 * the heat balance while not locking the load data during the whole
		 * balance step, which could cause all the workers to wait on the lock
		 * to update the data. */
		struct starpu_data_interface_ops *itf_load_data = starpu_data_get_interface_ops(*load_data_handle);
		void* itf_src = starpu_data_get_interface_on_node(*load_data_handle, STARPU_MAIN_RAM);
		void* itf_dst = starpu_data_get_interface_on_node(*load_data_handle_cpy, STARPU_MAIN_RAM);
		memcpy(itf_dst, itf_src, itf_load_data->interface_size);

		_STARPU_DEBUG("[node %d] Balance phase %d\n", my_rank, load_data_get_current_phase(*load_data_handle));
		STARPU_PTHREAD_MUTEX_UNLOCK(&load_data_mutex);

		heat_balance(*load_data_handle_cpy);
	}
}

static void finished_task_heat()
{
	//fprintf(stderr,"Try to decrement nsubmitted_tasks...");
	STARPU_PTHREAD_MUTEX_LOCK(&load_data_mutex);

	load_data_inc_nfinished_tasks(*load_data_handle);
	//fprintf(stderr,"Decrement nsubmitted_tasks, now %d\n", load_data_get_nsubmitted_tasks(*load_data_handle));
	if (load_data_wakeup_cond(*load_data_handle))
	{
		//fprintf(stderr,"Wakeup ! nfinished_tasks = %d, wakeup_threshold = %d\n", load_data_get_nfinished_tasks(*load_data_handle), load_data_get_wakeup_threshold(*load_data_handle));
		load_data_update_elapsed_time(*load_data_handle);
		STARPU_PTHREAD_COND_SIGNAL(&load_data_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&load_data_mutex);
	}
	else
		STARPU_PTHREAD_MUTEX_UNLOCK(&load_data_mutex);
}

/******************************************************************************
 *                  Initialization / Deinitialization                         *
 *****************************************************************************/

static int init_heat(struct starpu_mpi_lb_conf *itf)
{
	int i;
	int sleep_task_threshold;
	double wakeup_ratio;

	starpu_mpi_comm_size(MPI_COMM_WORLD, &world_size);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);

	/* Immediately return if the starpu_mpi_lb_conf is invalid. */
	if (!(itf && itf->get_neighbors && itf->get_data_unit_to_migrate))
	{
		_STARPU_MSG("Error: struct starpu_mpi_lb_conf %p invalid\n", itf);
		return 1;
	}

	_STARPU_MPI_MALLOC(user_itf, sizeof(struct starpu_mpi_lb_conf));
	memcpy(user_itf, itf, sizeof(struct starpu_mpi_lb_conf));

	/* Get the neighbors of the local MPI node */
	user_itf->get_neighbors(&neighbor_ids, &nneighbors);
	if (nneighbors == 0)
	{
		_STARPU_MSG("Error: Function get_neighbors returning 0 neighbor\n");
		free(user_itf);
		user_itf = NULL;
		return 2;
	}

	/* The sleep threshold is deducted from the numbering of tasks by the
	 * application. For example, with this threshold, the submission thread
	 * will stop when a task for which the numbering is 2000 or above will be
	 * submitted to StarPU-MPI. However, much less tasks can be really
	 * submitted to the local MPI node: the sleeping of the submission threads
	 * checks the numbering of the tasks, not how many tasks have been
	 * submitted to the local MPI node, which are two different things. */
	char *sleep_env = starpu_getenv("LB_HEAT_SLEEP_THRESHOLD");
	if (sleep_env)
		sleep_task_threshold = atoi(sleep_env);
	else
		sleep_task_threshold = 2000;

	char *wakeup_env = starpu_getenv("LB_HEAT_WAKEUP_RATIO");
	if (wakeup_env)
		wakeup_ratio = atof(wakeup_env);
	else
		wakeup_ratio = 0.5;

	char *time_env = starpu_getenv("LB_HEAT_TIME_THRESHOLD");
	if (time_env)
		time_threshold = atoi(time_env);
	else
		time_threshold = 2000;

	STARPU_PTHREAD_MUTEX_INIT(&load_data_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&load_data_cond, NULL);

	/* Allocate, initialize and register all the data handles that will be
	 * needed for the load balancer, to not reallocate them at each balance
	 * step. */

	/* Local load data */
	_STARPU_MPI_CALLOC(load_data_handle, 1, sizeof(starpu_data_handle_t));
	load_data_data_register(load_data_handle, STARPU_MAIN_RAM, sleep_task_threshold, wakeup_ratio);

	/* Copy of the local load data to enable parallel update of the load data
	 * with communications to neighbor nodes */
	_STARPU_MPI_CALLOC(load_data_handle_cpy, 1, sizeof(starpu_data_handle_t));
	void *local_interface = starpu_data_get_interface_on_node(*load_data_handle, STARPU_MAIN_RAM);
	struct starpu_data_interface_ops *itf_load_data = starpu_data_get_interface_ops(*load_data_handle);
	starpu_data_register(load_data_handle_cpy, STARPU_MAIN_RAM, local_interface, itf_load_data);
	starpu_mpi_data_register(*load_data_handle_cpy, TAG_LOAD(my_rank), my_rank);

	/* Remote load data */
	_STARPU_MPI_CALLOC(neighbor_load_data_handles, nneighbors, sizeof(starpu_data_handle_t));
	for (i = 0; i < nneighbors; i++)
	{
		load_data_data_register(&neighbor_load_data_handles[i], STARPU_MAIN_RAM, sleep_task_threshold, wakeup_ratio);
		starpu_mpi_data_register(neighbor_load_data_handles[i], TAG_LOAD(neighbor_ids[i]), neighbor_ids[i]);
	}

	/* Data movements handles */
	_STARPU_MPI_MALLOC(data_movements_handles, world_size*sizeof(starpu_data_handle_t));
	for (i = 0; i < world_size; i++)
	{
		data_movements_data_register(&data_movements_handles[i], STARPU_MAIN_RAM, NULL, NULL, 0);
		starpu_mpi_data_register(data_movements_handles[i], TAG_MOV(i), i);
	}

	/* Hash table of moved data that will be brought back on the node at
	 * termination time */
	mdh = NULL;

	return 0;
}

/* Move back all the data that has been migrated out of this node at
 * denitialization time of the load balancer, to ensure the consistency with
 * the ranks of data originally registered by the application. */
static void move_back_data()
{
	int i,j;

	/* Update the new ranks for all concerned data */
	for (i = 0; i < world_size; i++)
	{
		/* In this case, each data_movements_handles contains the handles to move back on the specific node */
		int ndata_to_update = data_movements_get_size_tables(data_movements_handles[i]);
		if (ndata_to_update)
		{
			_STARPU_DEBUG("Move back %d data from table %d on node %d\n", ndata_to_update, i, my_rank);

			for (j = 0; j < ndata_to_update; j++)
			{
				starpu_data_handle_t handle = _starpu_mpi_tag_get_data_handle_from_tag((data_movements_get_tags_table(data_movements_handles[i]))[j]);
				STARPU_ASSERT(handle);

				int dst_rank = (data_movements_get_ranks_table(data_movements_handles[i]))[j];
				STARPU_ASSERT(i == dst_rank);

				if (i == my_rank)
				{
					/* The data is moved back, so update the state of the
					 * moved_data hash table to reflect this change */
					struct moved_data_entry *md = NULL;
					HASH_FIND_PTR(mdh, &handle, md);
					if (md)
					{
						HASH_DEL(mdh, md);
						free(md);
					}
				}

				//fprintf(stderr,"Call of starpu_mpi_get_data_on_node(%d,%d) on node %d\n", starpu_mpi_data_get_tag(handle), dst_rank, my_rank);

				/* Migrate the data handle */
				starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, handle, dst_rank, NULL, NULL);

				//fprintf(stderr,"New rank (%d) of data %d upgraded on node %d\n", dst_rank, starpu_mpi_data_get_tag(handle), my_rank);
				starpu_mpi_data_set_rank_comm(handle, dst_rank, MPI_COMM_WORLD);
			}
		}
	}
}

static int deinit_heat()
{
	int i;

	if ((!user_itf) || (nneighbors == 0))
		return 1;

	_STARPU_DEBUG("Shutting down heat lb policy\n");

	unsigned int ndata_to_move_back = HASH_COUNT(mdh);

	if (ndata_to_move_back)
	{
		_STARPU_DEBUG("Move back %u data on node %d ..\n", ndata_to_move_back, my_rank);
		data_movements_reallocate_tables(data_movements_handles[my_rank], ndata_to_move_back);

		starpu_mpi_tag_t *tags = data_movements_get_tags_table(data_movements_handles[my_rank]);
		int *ranks = data_movements_get_ranks_table(data_movements_handles[my_rank]);

		int n = 0;
		struct moved_data_entry *md=NULL, *tmp=NULL;
		HASH_ITER(hh, mdh, md, tmp)
		{
			tags[n] = starpu_mpi_data_get_tag(md->handle);
			ranks[n] = my_rank;
			n++;
		}
	}
	else
		data_movements_reallocate_tables(data_movements_handles[my_rank], 0);

	exchange_data_movements_infos();
	move_back_data();

	/* This assert ensures that all nodes have properly gotten back all the
	 * data that has been moven out of the node. */
	STARPU_ASSERT(HASH_COUNT(mdh) == 0);
	free(mdh);
	mdh = NULL;

	starpu_data_unregister(*load_data_handle);
	free(load_data_handle);
	load_data_handle = NULL;

	starpu_mpi_cache_flush(MPI_COMM_WORLD, *load_data_handle_cpy);
	starpu_data_unregister(*load_data_handle_cpy);
	free(load_data_handle_cpy);
	load_data_handle_cpy = NULL;

	for (i = 0; i < nneighbors; i++)
	{
		starpu_mpi_cache_flush(MPI_COMM_WORLD, neighbor_load_data_handles[i]);
		starpu_data_unregister(neighbor_load_data_handles[i]);
	}
	free(neighbor_load_data_handles);
	neighbor_load_data_handles = NULL;

	nneighbors = 0;
	free(neighbor_ids);
	neighbor_ids = NULL;

	for (i = 0; i < world_size; i++)
	{
		starpu_mpi_cache_flush(MPI_COMM_WORLD, data_movements_handles[i]);
		data_movements_reallocate_tables(data_movements_handles[i], 0);
		starpu_data_unregister(data_movements_handles[i]);
	}
	free(data_movements_handles);
	data_movements_handles = NULL;

	STARPU_PTHREAD_MUTEX_DESTROY(&load_data_mutex);
	STARPU_PTHREAD_COND_DESTROY(&load_data_cond);
	free(user_itf);
	user_itf = NULL;

	return 0;
}

/******************************************************************************
 *                                  Policy                                    *
 *****************************************************************************/

struct load_balancer_policy load_heat_propagation_policy =
{
	.init = init_heat,
	.deinit = deinit_heat,
	.submitted_task_entry_point = submitted_task_heat,
	.finished_task_entry_point = finished_task_heat,
	.policy_name = "heat"
};

#endif
