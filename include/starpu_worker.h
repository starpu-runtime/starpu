/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2017                                Inria
 * Copyright (C) 2010-2015,2017                           CNRS
 * Copyright (C) 2009-2014,2016,2017,2019                 Université de Bordeaux
 * Copyright (C) 2013                                     Thibaut Lambert
 * Copyright (C) 2016                                     Uppsala University
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

#ifndef __STARPU_WORKER_H__
#define __STARPU_WORKER_H__

#include <stdlib.h>
#include <starpu_config.h>
#include <starpu_thread.h>
#include <starpu_task.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

enum starpu_worker_archtype
{
	STARPU_CPU_WORKER,
	STARPU_CUDA_WORKER,
	STARPU_OPENCL_WORKER,
	STARPU_MIC_WORKER,
	STARPU_SCC_WORKER,
	STARPU_MPI_MS_WORKER,
	STARPU_ANY_WORKER
};

struct starpu_sched_ctx_iterator
{
	int cursor;
	void *value;
	void *possible_value;
	char visited[STARPU_NMAXWORKERS];
	int possibly_parallel; 
};

enum starpu_worker_collection_type
{
	STARPU_WORKER_TREE,
	STARPU_WORKER_LIST
};


struct starpu_worker_collection
{
	int *workerids;
	void *collection_private;
	unsigned nworkers;
	void *unblocked_workers;
	unsigned nunblocked_workers;
	void *masters;
	unsigned nmasters;
	char present[STARPU_NMAXWORKERS];
	char is_unblocked[STARPU_NMAXWORKERS];
	char is_master[STARPU_NMAXWORKERS];
	enum starpu_worker_collection_type type;
	unsigned (*has_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	int (*get_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	int (*add)(struct starpu_worker_collection *workers, int worker);
	int (*remove)(struct starpu_worker_collection *workers, int worker);
	void (*init)(struct starpu_worker_collection *workers);
	void (*deinit)(struct starpu_worker_collection *workers);
	void (*init_iterator)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	void (*init_iterator_for_parallel_tasks)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it, struct starpu_task *task);
};

extern struct starpu_worker_collection worker_list;
extern struct starpu_worker_collection worker_tree;

unsigned starpu_worker_get_count(void);
unsigned starpu_combined_worker_get_count(void);
unsigned starpu_worker_is_combined_worker(int id);

unsigned starpu_cpu_worker_get_count(void);
unsigned starpu_cuda_worker_get_count(void);
unsigned starpu_opencl_worker_get_count(void);
unsigned starpu_mic_worker_get_count(void);
unsigned starpu_scc_worker_get_count(void);
unsigned starpu_mpi_ms_worker_get_count(void);

unsigned starpu_mic_device_get_count(void);

int starpu_worker_get_id(void);
unsigned _starpu_worker_get_id_check(const char *f, int l);
unsigned starpu_worker_get_id_check(void);
#define starpu_worker_get_id_check() _starpu_worker_get_id_check(__FILE__, __LINE__)
int starpu_worker_get_bindid(int workerid);

int starpu_combined_worker_get_id(void);
int starpu_combined_worker_get_size(void);
int starpu_combined_worker_get_rank(void);

enum starpu_worker_archtype starpu_worker_get_type(int id);

int starpu_worker_get_count_by_type(enum starpu_worker_archtype type);

unsigned starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, unsigned maxsize);

int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);

int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);

void starpu_worker_get_name(int id, char *dst, size_t maxlen);

void starpu_worker_display_names(FILE *output, enum starpu_worker_archtype type);

int starpu_worker_get_devid(int id);

int starpu_worker_get_mp_nodeid(int id);

struct starpu_tree* starpu_workers_get_tree(void);

unsigned starpu_worker_get_sched_ctx_list(int worker, unsigned **sched_ctx);

unsigned starpu_worker_is_blocked_in_parallel(int workerid);

unsigned starpu_worker_is_slave_somewhere(int workerid);

char *starpu_worker_get_type_as_string(enum starpu_worker_archtype type);

int starpu_bindid_get_workerids(int bindid, int **workerids);

int starpu_worker_get_devids(enum starpu_worker_archtype type, int *devids, int num);

int starpu_worker_get_stream_workerids(unsigned devid, int *workerids, enum starpu_worker_archtype type);

unsigned starpu_worker_get_sched_ctx_id_stream(unsigned stream_workerid);

int starpu_worker_sched_op_pending(void);

void starpu_worker_relax_on(void);

void starpu_worker_relax_off(void);

int starpu_worker_get_relax_state(void);

void starpu_worker_lock(int workerid);

int starpu_worker_trylock(int workerid);

void starpu_worker_unlock(int workerid);

void starpu_worker_lock_self(void);

void starpu_worker_unlock_self(void);

int starpu_wake_worker_relax(int workerid);

#ifdef STARPU_WORKER_CALLBACKS
void starpu_worker_set_going_to_sleep_callback(void (*callback)(unsigned workerid));

void starpu_worker_set_waking_up_callback(void (*callback)(unsigned workerid));
#endif

#ifdef STARPU_HAVE_HWLOC
hwloc_cpuset_t starpu_worker_get_hwloc_cpuset(int workerid);
hwloc_obj_t starpu_worker_get_hwloc_obj(int workerid);
#endif

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_WORKER_H__ */

