/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
	STARPU_ANY_WORKER
};

struct starpu_sched_ctx_iterator
{
	int cursor;
	void *value;
	void *possible_value;
	char visited[STARPU_NMAXWORKERS];
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
	void *masters;
	unsigned nmasters;
	char present[STARPU_NMAXWORKERS];
	char is_master[STARPU_NMAXWORKERS];
	enum starpu_worker_collection_type type;
	unsigned (*has_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	int (*get_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	unsigned (*has_next_master)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	int (*get_next_master)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	int (*add)(struct starpu_worker_collection *workers, int worker);
	int (*remove)(struct starpu_worker_collection *workers, int worker);
	void (*init)(struct starpu_worker_collection *workers);
	void (*deinit)(struct starpu_worker_collection *workers);
	void (*init_iterator)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
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

int starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);

int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);

int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);

void starpu_worker_get_name(int id, char *dst, size_t maxlen);

int starpu_worker_get_devid(int id);

int starpu_worker_get_mp_nodeid(int id);

struct starpu_tree* starpu_workers_get_tree(void);

unsigned starpu_worker_get_sched_ctx_list(int worker, unsigned **sched_ctx);

unsigned starpu_worker_is_slave(int workerid);

char *starpu_worker_get_type_as_string(enum starpu_worker_archtype type);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_WORKER_H__ */

