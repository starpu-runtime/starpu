/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2018                                Université de Bordeaux
 * Copyright (C) 2011-2013,2016,2017                      Inria
 * Copyright (C) 2010-2015,2017                           CNRS
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

#ifndef __STARPU_DATA_H__
#define __STARPU_DATA_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct _starpu_data_state;
typedef struct _starpu_data_state* starpu_data_handle_t;

/* Note: when adding a flag here, update _starpu_detect_implicit_data_deps_with_handle */
enum starpu_data_access_mode
{
	STARPU_NONE=0,
	STARPU_R=(1<<0),
	STARPU_W=(1<<1),
	STARPU_RW=(STARPU_R|STARPU_W),
	STARPU_SCRATCH=(1<<2),
	STARPU_REDUX=(1<<3),
	STARPU_COMMUTE=(1<<4),
	STARPU_SSEND=(1<<5),
	STARPU_LOCALITY=(1<<6),
	STARPU_UNMAP=(1<<7),
	STARPU_ACCESS_MODE_MAX=(1<<8)
	/* Note: other STARPU_* values in include/starpu_task_util.h */
};

struct starpu_data_descr
{
	starpu_data_handle_t handle;
	enum starpu_data_access_mode mode;
};

struct starpu_data_interface_ops;

void starpu_data_set_name(starpu_data_handle_t handle, const char *name);
void starpu_data_set_coordinates_array(starpu_data_handle_t handle, int dimensions, int dims[]);
void starpu_data_set_coordinates(starpu_data_handle_t handle, unsigned dimensions, ...);

void starpu_data_unregister(starpu_data_handle_t handle);
void starpu_data_unregister_no_coherency(starpu_data_handle_t handle);
void starpu_data_unregister_submit(starpu_data_handle_t handle);
void starpu_data_invalidate(starpu_data_handle_t handle);
void starpu_data_invalidate_submit(starpu_data_handle_t handle);

void starpu_data_advise_as_important(starpu_data_handle_t handle, unsigned is_important);

#define STARPU_ACQUIRE_NO_NODE -1
#define STARPU_ACQUIRE_NO_NODE_LOCK_ALL -2
int starpu_data_acquire(starpu_data_handle_t handle, enum starpu_data_access_mode mode);
int starpu_data_acquire_on_node(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode);
int starpu_data_acquire_cb(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg);
int starpu_data_acquire_on_node_cb(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg);
int starpu_data_acquire_cb_sequential_consistency(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);
int starpu_data_acquire_on_node_cb_sequential_consistency(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);
int starpu_data_acquire_on_node_cb_sequential_consistency_quick(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency, int quick);
int starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency, int quick, long *pre_sync_jobid, long *post_sync_jobid);

int starpu_data_acquire_try(starpu_data_handle_t handle, enum starpu_data_access_mode mode);
int starpu_data_acquire_on_node_try(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode);

#ifdef __GCC__
#  define STARPU_DATA_ACQUIRE_CB(handle, mode, code) do \
	{ \						\
		void callback(void *arg)		\
		{					\
			code;				\
			starpu_data_release(handle);  	\
		}			      		\
		starpu_data_acquire_cb(handle, mode, callback, NULL);	\
	}						\
	while(0)
#endif

void starpu_data_release(starpu_data_handle_t handle);
void starpu_data_release_on_node(starpu_data_handle_t handle, int node);

typedef struct starpu_arbiter *starpu_arbiter_t;
starpu_arbiter_t starpu_arbiter_create(void) STARPU_ATTRIBUTE_MALLOC;
void starpu_data_assign_arbiter(starpu_data_handle_t handle, starpu_arbiter_t arbiter);
void starpu_arbiter_destroy(starpu_arbiter_t arbiter);

void starpu_data_display_memory_stats();

#define starpu_data_malloc_pinned_if_possible	starpu_malloc
#define starpu_data_free_pinned_if_possible	starpu_free

int starpu_data_request_allocation(starpu_data_handle_t handle, unsigned node);

int starpu_data_fetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);
int starpu_data_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);
int starpu_data_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);
int starpu_data_idle_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);
int starpu_data_idle_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);

void starpu_data_wont_use(starpu_data_handle_t handle);

#define STARPU_MAIN_RAM 0

enum starpu_node_kind
{
	STARPU_UNUSED     = 0x00,
	STARPU_CPU_RAM    = 0x01,
	STARPU_CUDA_RAM   = 0x02,
	STARPU_OPENCL_RAM = 0x03,
	STARPU_DISK_RAM   = 0x04,
	STARPU_MIC_RAM    = 0x05,
	STARPU_SCC_RAM    = 0x06,
	STARPU_SCC_SHM    = 0x07,
	STARPU_MPI_MS_RAM = 0x08

};

unsigned starpu_worker_get_memory_node(unsigned workerid);
unsigned starpu_memory_nodes_get_count(void);
int starpu_memory_nodes_get_numa_count(void);
int starpu_memory_nodes_numa_id_to_devid(int osid);
int starpu_memory_nodes_numa_devid_to_id(unsigned id);

enum starpu_node_kind starpu_node_get_kind(unsigned node);

void starpu_data_set_wt_mask(starpu_data_handle_t handle, uint32_t wt_mask);

void starpu_data_set_sequential_consistency_flag(starpu_data_handle_t handle, unsigned flag);
unsigned starpu_data_get_sequential_consistency_flag(starpu_data_handle_t handle);
unsigned starpu_data_get_default_sequential_consistency_flag(void);
void starpu_data_set_default_sequential_consistency_flag(unsigned flag);

void starpu_data_query_status(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_requested);

struct starpu_codelet;

void starpu_data_set_reduction_methods(starpu_data_handle_t handle, struct starpu_codelet *redux_cl, struct starpu_codelet *init_cl);

struct starpu_data_interface_ops* starpu_data_get_interface_ops(starpu_data_handle_t handle);

unsigned starpu_data_test_if_allocated_on_node(starpu_data_handle_t handle, unsigned memory_node);

void starpu_memchunk_tidy(unsigned memory_node);

void starpu_data_set_user_data(starpu_data_handle_t handle, void* user_data);
void *starpu_data_get_user_data(starpu_data_handle_t handle);

int starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle, int asynchronous, void (*callback_func)(void*), void *callback_arg);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DATA_H__ */
