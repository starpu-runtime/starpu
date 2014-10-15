/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2014  Université de Bordeaux
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

void _starpu_simgrid_init(void);
void _starpu_simgrid_execute_job(struct _starpu_job *job, struct starpu_perfmodel_arch* perf_arch, double length);
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req);
/* Return the number of hosts prefixed by PREFIX */
int _starpu_simgrid_get_nbhosts(const char *prefix);
unsigned long long _starpu_simgrid_get_memsize(const char *prefix, unsigned devid);
msg_host_t _starpu_simgrid_get_host_by_name(const char *name);
void _starpu_simgrid_get_platform_path(char *path, size_t maxlen);
#endif

#endif // __SIMGRID_H__
