/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <pthread.h>
#include <core/workers.h>
#include <datawizard/progress.h>
#include <datawizard/data_request.h>

extern pthread_key_t local_workers_key;

#ifdef USE_GORDON
extern void handle_terminated_job_per_worker(struct worker_s *worker);
extern pthread_spinlock_t terminated_list_mutexes[32]; 
#endif

void datawizard_progress(uint32_t memory_node)
{
	/* in case some other driver requested data */
	handle_node_data_requests(memory_node);

#ifdef USE_GORDON
	/* XXX quick and dirty !! */
	struct worker_set_s *set;
	set = pthread_getspecific(local_workers_key);
	if (set) {
		/* make the corresponding workers progress */
		unsigned worker;
		for (worker = 0; worker < set->nworkers; worker++)
		{
			pthread_spin_lock(&terminated_list_mutexes[0]);
			handle_terminated_job_per_worker(&set->workers[worker]);
			pthread_spin_unlock(&terminated_list_mutexes[0]);
		}
	}
#endif
}
