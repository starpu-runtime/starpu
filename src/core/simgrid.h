/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2015  Université de Bordeaux
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

#include <datawizard/data_request.h>

#ifdef STARPU_SIMGRID
#include <msg/msg.h>

struct _starpu_pthread_args
{
	void *(*f)(void*);
	void *arg;
};

#define MAX_TSD 16

#define STARPU_MPI_AS_PREFIX "StarPU-MPI"
#define _starpu_simgrid_running_smpi() (getenv("SMPI_GLOBAL_SIZE") != NULL)

void _starpu_simgrid_init(void);
void _starpu_simgrid_wait_tasks(int workerid);
void _starpu_simgrid_submit_job(int workerid, struct _starpu_job *job, struct starpu_perfmodel_arch* perf_arch, double length, unsigned *finished, starpu_pthread_mutex_t *mutex, starpu_pthread_cond_t *cond);
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req);
/* Return the number of hosts prefixed by PREFIX */
int _starpu_simgrid_get_nbhosts(const char *prefix);
unsigned long long _starpu_simgrid_get_memsize(const char *prefix, unsigned devid);
msg_host_t _starpu_simgrid_get_host_by_name(const char *name);
msg_host_t _starpu_simgrid_get_host_by_worker(struct _starpu_worker *worker);
void _starpu_simgrid_get_platform_path(char *path, size_t maxlen);
msg_as_t _starpu_simgrid_get_as_by_name(const char *name);
#pragma weak starpu_mpi_world_rank
extern int starpu_mpi_world_rank(void);
#endif

#endif // __SIMGRID_H__
