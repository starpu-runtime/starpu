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


#include <mpi.h>
#include <dlfcn.h>

#include "driver_mpi_sink.h"
#include "driver_mpi_source.h"
#include "driver_mpi_common.h"

void _starpu_mpi_sink_init(struct _starpu_mp_node *node)
{
        _starpu_mpi_common_mp_initialize_src_sink(node);

        _STARPU_MALLOC(node->thread_table, sizeof(starpu_pthread_t)*node->nb_cores);
        //TODO
}

void _starpu_mpi_sink_deinit(struct _starpu_mp_node *node)
{
	int i;
	node->is_running = 0;
	for(i=0; i<node->nb_cores; i++)
	{
		sem_post(&node->sem_run_table[i]);
		STARPU_PTHREAD_JOIN(((starpu_pthread_t *)node->thread_table)[i],NULL);
	}
        free(node->thread_table);
}

void (*_starpu_mpi_sink_lookup (const struct _starpu_mp_node * node STARPU_ATTRIBUTE_UNUSED, char* func_name))(void)
{
        void *dl_handle = dlopen(NULL, RTLD_NOW);
        return dlsym(dl_handle, func_name);
}

void _starpu_mpi_sink_launch_workers(struct _starpu_mp_node *node)
{
        //TODO
        int i;
        struct arg_sink_thread * arg;
        cpu_set_t cpuset;
        starpu_pthread_attr_t attr;
        starpu_pthread_t thread;

        for(i=0; i < node->nb_cores; i++)
        {
		int ret;

                //init the set
                CPU_ZERO(&cpuset);
                CPU_SET(i,&cpuset);

                ret = starpu_pthread_attr_init(&attr);
                STARPU_ASSERT(ret == 0);
                ret = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
                STARPU_ASSERT(ret == 0);

                /*prepare the argument for the thread*/
                _STARPU_MALLOC(arg, sizeof(struct arg_sink_thread));
                arg->coreid = i;
                arg->node = node;

                STARPU_PTHREAD_CREATE(&thread, &attr, _starpu_sink_thread, arg);
                ((starpu_pthread_t *)node->thread_table)[i] = thread;

        }
}

void _starpu_mpi_sink_bind_thread(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, int coreid, int * core_table, int nb_core)
{
        //TODO
}
