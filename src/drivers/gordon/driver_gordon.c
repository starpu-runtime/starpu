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

#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <semaphore.h>
#include "driver_gordon.h"
#include "gordon_interface.h"
#include <core/policies/sched_policy.h>

pthread_t progress_thread;
sem_t progress_sem;
struct starpu_mutex_t terminated_list_mutexes[32]; 

struct gordon_task_wrapper_s {
	/* who has executed that ? */
	struct worker_s *worker;

	struct job_list_s *list;	/* StarPU */
	struct gordon_ppu_job_s *gordon_job; /* gordon*/

	struct job_s *j; /* if there is a single task */

	/* debug */
	unsigned terminated;
};

void *gordon_worker_progress(void *arg)
{
	fprintf(stderr, "gordon_worker_progress\n");

#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	struct worker_set_s *gordon_set_arg = arg;
	unsigned prog_thread_bind_id = 
		(gordon_set_arg->workers[0].bindid + 1)%(sysconf(_SC_NPROCESSORS_ONLN));
	cpu_set_t aff_mask; 
	CPU_ZERO(&aff_mask);
	CPU_SET(prog_thread_bind_id, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

	sem_post(&progress_sem);

	while (1) {
		/* the Gordon runtime needs to make sure that we poll it 
		 * so that we handle jobs that are done */

		/* wait for one task termination */
		int ret = gordon_wait(0);
		if (ret)
		{
			/* possibly wake the thread that injects work */
			wake_all_blocked_workers();
		}
	}

	return NULL;
}

static void starpu_to_gordon_buffers(job_t j, struct gordon_ppu_job_s *gordon_job, uint32_t memory_node)
{
	unsigned buffer;
	unsigned nin = 0, ninout = 0, nout = 0;
	unsigned in = 0, inout = 0, out = 0;

	/* if it is non null, the argument buffer is considered
 	 * as the first read-only buffer */
	if (j->cl_arg) {
		gordon_job->buffers[in] = (uint64_t)j->cl_arg;
		gordon_job->ss[in].size = (uint32_t)j->cl_arg_size;
		
		nin++; in++;
	}

	/* count the number of in/inout/out buffers */
	for (buffer = 0; buffer < j->nbuffers; buffer++)
	{
		struct starpu_buffer_descr_t *descr;
		descr = &j->buffers[buffer];

		switch (descr->mode) {
			case R:
				nin++;
				break;
			case W:
				nout++;
				break;
			case RW:
			default:
				ninout++;
				break;
		}
	}

	for (buffer = 0; buffer < j->nbuffers; buffer++)
	{
		unsigned gordon_buffer;
		struct starpu_buffer_descr_t *descr;
		descr = &j->buffers[buffer];

		switch (descr->mode) {
			case R:
				gordon_buffer = in++;
				break;
			case W:
				gordon_buffer = nin + ninout + out++;
				break;
			case RW:
			default:
				gordon_buffer = nin + inout++;
				break;
		}

		struct starpu_data_state_t *state = j->buffers[buffer].state;

		gordon_job->nalloc = 0;
		gordon_job->nin = nin;
		gordon_job->ninout = ninout;
		gordon_job->nout = nout;

		STARPU_ASSERT(state->ops->convert_to_gordon);
		state->ops->convert_to_gordon(&state->interface[memory_node],
				&gordon_job->buffers[gordon_buffer],
				&gordon_job->ss[gordon_buffer]);
	}
}

/* we assume the data are already available so that the data interface fields are 
 * already filled */
static struct gordon_task_wrapper_s *starpu_to_gordon_job(job_t j)
{
	struct gordon_ppu_job_s *gordon_job = gordon_alloc_jobs(1, 0);
	struct gordon_task_wrapper_s *task_wrapper =
				malloc(sizeof(struct gordon_task_wrapper_s));

	task_wrapper->gordon_job = gordon_job;
	task_wrapper->j = j;
	task_wrapper->terminated = 0;

	gordon_job->index = j->cl->gordon_func;

	/* we should not hardcore the memory node ... XXX */
	unsigned memory_node = 0;
	starpu_to_gordon_buffers(j, gordon_job, memory_node);

	return task_wrapper;
}

void handle_terminated_job(job_t j)
{
	push_codelet_output(j->buffers, j->nbuffers, 0);
	handle_job_termination(j);
	wake_all_blocked_workers();
}

void handle_terminated_job_per_worker(struct worker_s *worker)
{

	if (STARPU_UNLIKELY(!worker->worker_is_running))
		return;

//	fprintf(stderr, " handle_terminated_job_per_worker worker %p worker->terminated_jobs %p \n", worker, worker->terminated_jobs);

	while (!job_list_empty(worker->terminated_jobs))
	{
		job_t j;
		j = job_list_pop_front(worker->terminated_jobs);
//		fprintf(stderr, "handle_terminated_job %p\n", j);
		handle_terminated_job(j);
	}
}

static void handle_terminated_jobs(struct worker_set_s *arg)
{
//	fprintf(stderr, "handle_terminated_jobs\n");

	/* terminate all the pending jobs and remove 
 	 * them from the terminated_jobs lists */
	unsigned spu;
	for (spu = 0; spu < arg->nworkers; spu++)
	{
		take_mutex(&terminated_list_mutexes[spu]);
		handle_terminated_job_per_worker(&arg->workers[spu]);
		release_mutex(&terminated_list_mutexes[spu]);
		//if (!take_mutex_try(&terminated_list_mutexes[spu]))
		//{
		//	handle_terminated_job_per_worker(&arg->workers[spu]);
		//	release_mutex(&terminated_list_mutexes[spu]);
		//}
	}
}

static void gordon_callback_list_func(void *arg)
{
	struct gordon_task_wrapper_s *task_wrapper = arg; 
	struct job_list_s *wrapper_list; 
	struct job_list_s *terminated_list; 

	/* we don't know who will execute that codelet : so we actually defer the
 	 * execution of the StarPU codelet and the job termination later */
	struct worker_s *worker = task_wrapper->worker;
	STARPU_ASSERT(worker);

	wrapper_list = task_wrapper->list;
	terminated_list = worker->terminated_jobs;

	task_wrapper->terminated = 1;

//	fprintf(stderr, "gordon callback : push job j %p\n", task_wrapper->j);

	/* XXX 0 was hardcoded */
	take_mutex(&terminated_list_mutexes[0]);
	while (!job_list_empty(wrapper_list))
	{
		job_t j = job_list_pop_back(wrapper_list);
		job_list_push_back(terminated_list, j);
	}

	/* the job list was allocated by the gordon driver itself */
	job_list_delete(wrapper_list);

	release_mutex(&terminated_list_mutexes[0]);

	wake_all_blocked_workers();
	free(task_wrapper->gordon_job);
	free(task_wrapper);
}


static void gordon_callback_func(void *arg)
{
	struct gordon_task_wrapper_s *task_wrapper = arg; 

	/* we don't know who will execute that codelet : so we actually defer the
 	 * execution of the StarPU codelet and the job termination later */
	struct worker_s *worker = task_wrapper->worker;
	STARPU_ASSERT(worker);

	task_wrapper->terminated = 1;

//	fprintf(stderr, "gordon callback : push job j %p\n", task_wrapper->j);

	/* XXX 0 was hardcoded */
	take_mutex(&terminated_list_mutexes[0]);
	job_list_push_back(worker->terminated_jobs, task_wrapper->j);
	release_mutex(&terminated_list_mutexes[0]);
	wake_all_blocked_workers();
	free(task_wrapper);
}

int inject_task(job_t j, struct worker_s *worker)
{
	int ret = fetch_codelet_input(j->buffers, j->interface, j->nbuffers, 0);

	if (ret != 0) {
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return STARPU_TRYAGAIN;
	}

	struct gordon_task_wrapper_s *task_wrapper = starpu_to_gordon_job(j);

	task_wrapper->worker = worker;

	gordon_pushjob(task_wrapper->gordon_job, gordon_callback_func, task_wrapper);

	return 0;
}

int inject_task_list(struct job_list_s *list, struct worker_s *worker)
{
	/* first put back all tasks that can not be performed by Gordon */
	unsigned nvalids = 0;
	unsigned ninvalids = 0;
	job_t j;

	// TODO !
//	
//	for (j = job_list_begin(list); j != job_list_end(list); j = job_list_next(j) )
//	{
//		if (!GORDON_MAY_PERFORM(j)) {
//			// XXX TODO
//			ninvalids++;
//			assert(0);
//		}
//		else {
//			nvalids++;
//		}
//	}

	nvalids = job_list_size(list);
//	fprintf(stderr, "nvalids %d \n", nvalids);

	struct gordon_task_wrapper_s *task_wrapper = malloc(sizeof(struct gordon_task_wrapper_s));
	gordon_job_t *gordon_jobs = gordon_alloc_jobs(nvalids, 0);

	task_wrapper->gordon_job = gordon_jobs;
	task_wrapper->list = list;
	task_wrapper->j = NULL;
	task_wrapper->terminated = 0;
	task_wrapper->worker = worker;
	
	unsigned index;
	for (j = job_list_begin(list), index = 0; j != job_list_end(list); j = job_list_next(j), index++)
	{
		int ret;

		ret = fetch_codelet_input(j->buffers, j->interface, j->nbuffers, 0);
		STARPU_ASSERT(!ret);

		gordon_jobs[index].index = j->cl->gordon_func;

		/* we should not hardcore the memory node ... XXX */
		unsigned memory_node = 0;
		starpu_to_gordon_buffers(j, &gordon_jobs[index], memory_node);
		
	}

	gordon_pushjob(task_wrapper->gordon_job, gordon_callback_list_func, task_wrapper);

	return 0;
}

void *gordon_worker_inject(struct worker_set_s *arg)
{

	while(machine_is_running()) {
		/* make gordon driver progress */
		handle_terminated_jobs(arg);

		if (gordon_busy_enough()) {
			/* gordon already has enough work, wait a little TODO */
			wait_on_sched_event();
		}
		else {
#ifndef NOCHAIN
			int ret = 0;
			struct job_list_s *list = pop_every_task();
			/* XXX 0 is hardcoded */
			if (list)
			{
				/* partition lists */
				unsigned size = job_list_size(list);
				unsigned nchunks = (size<2*arg->nworkers)?size:(2*arg->nworkers);

				/* last element may be a little smaller (by 1) */
				unsigned chunksize = size/nchunks;

				unsigned chunk;
				for (chunk = 0; chunk < nchunks; chunk++)
				{
					struct job_list_s *chunk_list;
					if (chunk != (nchunks -1))
					{
						/* split the list in 2 parts : list = chunk_list | tail */
						chunk_list = job_list_new();

						/* find the end */
						chunk_list->_head = list->_head;

						job_itor_t it_j = job_list_begin(list);
						unsigned ind;
						for (ind = 0; ind < chunksize; ind++)
						{
							it_j = job_list_next(it_j);
						}

						/* it_j should be the first element of the new list (tail) */
						chunk_list->_tail = it_j->_prev;
						chunk_list->_tail->_next = NULL;
						list->_head = it_j;
						it_j->_prev = NULL;
					}
					else {
						/* this is the last chunk */
						chunk_list = list;
					}

					ret = inject_task_list(chunk_list, &arg->workers[0]);
				}
			}
			else {
				wait_on_sched_event();
			}
#else
			/* gordon should accept a little more work */
			job_t j;
			j =  pop_task();
	//		fprintf(stderr, "pop task %p\n", j);
			if (j) {
				if (GORDON_MAY_PERFORM(j)) {
					/* inject that task */
					/* XXX we hardcore &arg->workers[0] for now */
					inject_task(j, &arg->workers[0]);
				}
				else {
					push_task(j);
				}
			}
#endif
			
		}
	}

	return NULL;
}

extern pthread_key_t local_workers_key;

void *gordon_worker(void *arg)
{
	struct worker_set_s *gordon_set_arg = arg;

#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask; 
	CPU_ZERO(&aff_mask);
	CPU_SET(gordon_set_arg->workers[0].bindid, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif


	/* TODO set_local_memory_node per SPU */
	gordon_init(gordon_set_arg->nworkers);	

	/* XXX quick and dirty ... */
	pthread_setspecific(local_workers_key, arg);

	unsigned spu;
	for (spu = 0; spu < gordon_set_arg->nworkers; spu++)
	{
		init_mutex(&terminated_list_mutexes[spu]);
	}


	/*
 	 * To take advantage of PPE being hyperthreaded, we should have 2 threads
 	 * for the gordon driver : one injects works, the other makes sure that
 	 * gordon is progressing (and performs the callbacks).
	 */

	/* launch the progression thread */
	sem_init(&progress_sem, 0, 0);
	pthread_create(&progress_thread, NULL, gordon_worker_progress, gordon_set_arg);

	/* wait for the progression thread to be ready */
	sem_wait(&progress_sem);

	fprintf(stderr, "progress thread is running ... \n");
	
	/* tell the core that gordon is ready */
	sem_post(&gordon_set_arg->ready_sem);

	gordon_worker_inject(gordon_set_arg);

	fprintf(stderr, "gordon deinit...\n");
	gordon_deinit();
	fprintf(stderr, "gordon was deinited\n");

	pthread_exit((void *)0x42);
}
