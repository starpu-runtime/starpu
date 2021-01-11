/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __SIMGRID_H__
#define __SIMGRID_H__

#ifdef STARPU_SIMGRID
#ifdef STARPU_HAVE_SIMGRID_MSG_H
#include <simgrid/msg.h>
#else
#include <msg/msg.h>
#endif

#include <datawizard/data_request.h>

struct _starpu_pthread_args
{
	void *(*f)(void*);
	void *arg;
};

#define MAX_TSD 16

#define STARPU_MPI_AS_PREFIX "StarPU-MPI"
#define _starpu_simgrid_running_smpi() (getenv("SMPI_GLOBAL_SIZE") != NULL)

void _starpu_simgrid_init(int *argc, char ***argv);
void _starpu_simgrid_deinit(void);
void _starpu_simgrid_wait_tasks(int workerid);
void _starpu_simgrid_submit_job(int workerid, struct _starpu_job *job, struct starpu_perfmodel_arch* perf_arch, double length, unsigned *finished, starpu_pthread_mutex_t *mutex, starpu_pthread_cond_t *cond);
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req);
/* Return the number of hosts prefixed by PREFIX */
int _starpu_simgrid_get_nbhosts(const char *prefix);
unsigned long long _starpu_simgrid_get_memsize(const char *prefix, unsigned devid);
msg_host_t _starpu_simgrid_get_host_by_name(const char *name);
struct _starpu_worker;
msg_host_t _starpu_simgrid_get_host_by_worker(struct _starpu_worker *worker);
void _starpu_simgrid_get_platform_path(int version, char *path, size_t maxlen);
msg_as_t _starpu_simgrid_get_as_by_name(const char *name);
#pragma weak starpu_mpi_world_rank
extern int starpu_mpi_world_rank(void);
#pragma weak _starpu_mpi_simgrid_init
int _starpu_mpi_simgrid_init(int argc, char *argv[]);

#define _starpu_simgrid_cuda_malloc_cost() starpu_get_env_number_default("STARPU_SIMGRID_CUDA_MALLOC_COST", 1)
#define _starpu_simgrid_queue_malloc_cost() starpu_get_env_number_default("STARPU_SIMGRID_QUEUE_MALLOC_COST", 1)

#endif

#endif // __SIMGRID_H__
