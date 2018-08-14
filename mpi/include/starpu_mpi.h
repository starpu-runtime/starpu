/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2012,2014-2018                      Université de Bordeaux
 * Copyright (C) 2010-2018                                CNRS
 * Copyright (C) 2016                                     Inria
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

#ifndef __STARPU_MPI_H__
#define __STARPU_MPI_H__

#include <starpu.h>

#if defined(STARPU_USE_MPI)

#include <mpi.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef void *starpu_mpi_req;

typedef int64_t starpu_mpi_tag_t;

int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);
int starpu_mpi_isend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);
int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *req, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm);
int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);
int starpu_mpi_send_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);
int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, MPI_Status *status);
int starpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
int starpu_mpi_isend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg);
int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);
int starpu_mpi_issend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);
int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
int starpu_mpi_issend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg);
int starpu_mpi_wait(starpu_mpi_req *req, MPI_Status *status);
int starpu_mpi_test(starpu_mpi_req *req, int *flag, MPI_Status *status);
int starpu_mpi_barrier(MPI_Comm comm);

int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency);

int starpu_mpi_init_conf(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm, struct starpu_conf *conf);
int starpu_mpi_init_comm(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm);
int starpu_mpi_init(int *argc, char ***argv, int initialize_mpi);
int starpu_mpi_initialize(void) STARPU_DEPRECATED;
int starpu_mpi_initialize_extended(int *rank, int *world_size) STARPU_DEPRECATED;
int starpu_mpi_shutdown(void);

struct starpu_task *starpu_mpi_task_build(MPI_Comm comm, struct starpu_codelet *codelet, ...);
int starpu_mpi_task_post_build(MPI_Comm comm, struct starpu_codelet *codelet, ...);
int starpu_mpi_task_insert(MPI_Comm comm, struct starpu_codelet *codelet, ...);
/* the function starpu_mpi_insert_task has the same semantics as starpu_mpi_task_insert, it is kept to avoid breaking old codes */
int starpu_mpi_insert_task(MPI_Comm comm, struct starpu_codelet *codelet, ...);

void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node);
void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg);
void starpu_mpi_get_data_on_all_nodes_detached(MPI_Comm comm, starpu_data_handle_t data_handle);
void starpu_mpi_redux_data(MPI_Comm comm, starpu_data_handle_t data_handle);
void starpu_mpi_redux_data_prio(MPI_Comm comm, starpu_data_handle_t data_handle, int prio);

int starpu_mpi_scatter_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);
int starpu_mpi_gather_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);

int starpu_mpi_isend_detached_unlock_tag(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag);
int starpu_mpi_isend_detached_unlock_tag_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, starpu_tag_t tag);
int starpu_mpi_irecv_detached_unlock_tag(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag);

int starpu_mpi_isend_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, starpu_mpi_tag_t *data_tag, MPI_Comm *comm, starpu_tag_t tag);
int starpu_mpi_isend_array_detached_unlock_tag_prio(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, starpu_mpi_tag_t *data_tag, int *prio, MPI_Comm *comm, starpu_tag_t tag);
int starpu_mpi_irecv_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *source, starpu_mpi_tag_t *data_tag, MPI_Comm *comm, starpu_tag_t tag);

void starpu_mpi_comm_amounts_retrieve(size_t *comm_amounts);

void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle);
void starpu_mpi_cache_flush_all_data(MPI_Comm comm);

int starpu_mpi_cached_receive(starpu_data_handle_t data_handle);
int starpu_mpi_cached_send(starpu_data_handle_t data_handle, int dest);

int starpu_mpi_comm_size(MPI_Comm comm, int *size);
int starpu_mpi_comm_rank(MPI_Comm comm, int *rank);
int starpu_mpi_world_rank(void);
int starpu_mpi_world_size(void);

int starpu_mpi_get_communication_tag(void);
void starpu_mpi_set_communication_tag(int tag);

void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, starpu_mpi_tag_t data_tag, int rank, MPI_Comm comm);
#define starpu_mpi_data_register(data_handle, data_tag, rank) starpu_mpi_data_register_comm(data_handle, data_tag, rank, MPI_COMM_WORLD)

#define STARPU_MPI_PER_NODE -2

void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm);
#define starpu_mpi_data_set_rank(handle, rank) starpu_mpi_data_set_rank_comm(handle, rank, MPI_COMM_WORLD)
void starpu_mpi_data_set_tag(starpu_data_handle_t handle, starpu_mpi_tag_t data_tag);
#define starpu_data_set_rank starpu_mpi_data_set_rank
#define starpu_data_set_tag starpu_mpi_data_set_tag

int starpu_mpi_data_get_rank(starpu_data_handle_t handle);
starpu_mpi_tag_t starpu_mpi_data_get_tag(starpu_data_handle_t handle);
#define starpu_data_get_rank starpu_mpi_data_get_rank
#define starpu_data_get_tag starpu_mpi_data_get_tag

void starpu_mpi_data_migrate(MPI_Comm comm, starpu_data_handle_t handle, int new_rank);

#define STARPU_MPI_NODE_SELECTION_CURRENT_POLICY -1
#define STARPU_MPI_NODE_SELECTION_MOST_R_DATA    0

typedef int (*starpu_mpi_select_node_policy_func_t)(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data);
int starpu_mpi_node_selection_register_policy(starpu_mpi_select_node_policy_func_t policy_func);
int starpu_mpi_node_selection_unregister_policy(int policy);

int starpu_mpi_node_selection_get_current_policy();
int starpu_mpi_node_selection_set_current_policy(int policy);

int starpu_mpi_cache_is_enabled();
int starpu_mpi_cache_set(int enabled);

int starpu_mpi_wait_for_all(MPI_Comm comm);

typedef void (*starpu_mpi_datatype_allocate_func_t)(starpu_data_handle_t, MPI_Datatype *);
typedef void (*starpu_mpi_datatype_free_func_t)(MPI_Datatype *);
int starpu_mpi_datatype_register(starpu_data_handle_t handle, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func);
int starpu_mpi_datatype_unregister(starpu_data_handle_t handle);

int starpu_mpi_pre_submit_hook_register(void (*f)(struct starpu_task *));
int starpu_mpi_pre_submit_hook_unregister();

#define STARPU_MPI_TAG_UB MPI_TAG_UB
int starpu_mpi_comm_get_attr(MPI_Comm comm, int keyval, void *attribute_val, int *flag);

#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_MPI
#endif // __STARPU_MPI_H__
