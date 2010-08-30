/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "queues.h"
#include <common/utils.h>

/*
 * There can be various queue designs
 * 	- trivial single list
 * 	- cilk-like 
 * 	- hierarchical (marcel-like)
 */

void _starpu_setup_queues(void (*init_queue_design)(void),
		  struct starpu_jobq_s *(*func_init_queue)(void), 
		  struct starpu_machine_config_s *config) 
{
	unsigned worker;

	if (init_queue_design)
		init_queue_design();

	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct  starpu_worker_s *workerarg = &config->workers[worker];
		
		if (func_init_queue)
			workerarg->jobq = func_init_queue();
	}
}

void _starpu_deinit_queues(void (*deinit_queue_design)(void),
		  void (*func_deinit_queue)(struct starpu_jobq_s *q), 
		  struct starpu_machine_config_s *config)
{
	unsigned worker;

	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct  starpu_worker_s *workerarg = &config->workers[worker];
		
		if (func_deinit_queue)
			func_deinit_queue(workerarg->jobq);
	}

	if (deinit_queue_design)
		deinit_queue_design();
}
