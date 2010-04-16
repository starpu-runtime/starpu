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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <pthread.h>
#include <semaphore.h>
#include <common/utils.h>
#include "driver_gordon.h"
#include "gordon_interface.h"
#include <core/policies/sched_policy.h>

static unsigned progress_thread_is_inited = 0;

pthread_t progress_thread;

pthread_cond_t progress_cond;
pthread_mutex_t progress_mutex;

struct gordon_task_wrapper_s {
	/* who has executed that ? */
	struct starpu_worker_s *worker;

	struct starpu_job_list_s *list;	/* StarPU */
	struct gordon_ppu_job_s *gordon_job; /* gordon*/

	struct starpu_job_s *j; /* if there is a single task */

	/* debug */
	unsigned terminated;
};

void *gordon_worker_progress(void *arg)
{
	fprintf(stderr, "gordon_worker_progress\n");

	/* fix the thread on the correct cpu */
	struct starpu_worker_set_s *gordon_set_arg = arg;
	unsigned prog_thread_bind_id = 
		(gordon_set_arg->workers[0].bindid + 1)%(gordon_set_arg->config->nhwcores);
	_starpu_bind_thread_on_cpu(gordon_set_arg->config, prog_thread_bind_id);

	PTHREAD_MUTEX_LOCK(&progress_mutex);
	progress_thread_is_inited = 1;
	pthread_cond_signal(&progress_cond);
	PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	while (1) {
		/* the Gordon runtime needs to make sure that we poll it 
		 * so that we handle jobs that are done */

		/* wait for one task termination */
		int ret = gordon_wait(0);
		if (ret)
		{
			/* possibly wake the thread that injects work */
			starpu_wake_all_blocked_workers();
		}
	}

	return NULL;
}

static void starpu_to_gordon_buffers(starpu_job_t j, struct gordon_ppu_job_s *gordon_job, uint32_t memory_node)
{
	unsigned buffer;
	unsigned nin = 0, ninout = 0, nout = 0;
	unsigned in = 0, inout = 0, out = 0;

	struct starpu_task *task = j->task;
	struct starpu_codelet_t *cl = task->cl;

	/* if it is non null, the argument buffer is considered
 	 * as the first read-only buffer */
	if (task->cl_arg) {
		gordon_job->buffers[in] = (uint64_t)task->cl_arg;
		gordon_job->ss[in].size = (uint32_t)task->cl_arg_size;
		
		nin++; in++;
	}

	/* count the number of in/inout/out buffers */
	unsigned nbuffers = cl->nbuffers;
	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		struct starpu_buffer_descr_t *descr;
		descr = &task->buffers[buffer];

		switch (descr->mode) {
			case STARPU_R:
				nin++;
				break;
			case STARPU_W:
				nout++;
				break;
			case STARPU_RW:
			default:
				ninout++;
				break;
		}
	}

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		unsigned gordon_buffer;
		struct starpu_buffer_descr_t *descr;
		descr = &task->buffers[buffer];

		switch (descr->mode) {
			case STARPU_R:
				gordon_buffer = in++;
				break;
			case STARPU_W:
				gordon_buffer = nin + ninout + out++;
				break;
			case STARPU_RW:
			default:
				gordon_buffer = nin + inout++;
				break;
		}

		starpu_data_handle handle = task->buffers[buffer].handle;

		gordon_job->nalloc = 0;
		gordon_job->nin = nin;
		gordon_job->ninout = ninout;
		gordon_job->nout = nout;

		STARPU_ASSERT(handle->ops->convert_to_gordon);
		handle->ops->convert_to_gordon(&handle->interface[memory_node],
				&gordon_job->buffers[gordon_buffer],
				&gordon_job->ss[gordon_buffer]);
	}
}

/* we assume the data are already available so that the data interface fields are 
 * already filled */
static struct gordon_task_wrapper_s *starpu_to_gordon_job(starpu_job_t j)
{
	struct gordon_ppu_job_s *gordon_job = gordon_alloc_jobs(1, 0);
	struct gordon_task_wrapper_s *task_wrapper =
				malloc(sizeof(struct gordon_task_wrapper_s));

	task_wrapper->gordon_job = gordon_job;
	task_wrapper->j = j;
	task_wrapper->terminated = 0;

	gordon_job->index = j->task->cl->gordon_func;

	/* we should not hardcore the memory node ... XXX */
	unsigned memory_node = 0;
	starpu_to_gordon_buffers(j, gordon_job, memory_node);

	return task_wrapper;
}

static void handle_terminated_job(starpu_job_t j)
{
	_starpu_push_task_output(j->task, 0);
	_starpu_handle_job_termination(j);
	starpu_wake_all_blocked_workers();
}

static void gordon_callback_list_func(void *arg)
{
	struct gordon_task_wrapper_s *task_wrapper = arg; 
	struct starpu_job_list_s *wrapper_list; 

	/* we don't know who will execute that codelet : so we actually defer the
 	 * execution of the StarPU codelet and the job termination later */
	struct starpu_worker_s *worker = task_wrapper->worker;
	STARPU_ASSERT(worker);

	wrapper_list = task_wrapper->list;

	task_wrapper->terminated = 1;

//	fprintf(stderr, "gordon callback : push job j %p\n", task_wrapper->j);

	unsigned task_cnt = 0;

	/* XXX 0 was hardcoded */
	while (!starpu_job_list_empty(wrapper_list))
	{
		starpu_job_t j = starpu_job_list_pop_back(wrapper_list);

		struct gordon_ppu_job_s * gordon_task = &task_wrapper->gordon_job[task_cnt];
		struct starpu_perfmodel_t *model = j->task->cl->model;
		if (model && model->benchmarking)
		{
			double measured = (double)gordon_task->measured;
			unsigned cpuid = 0; /* XXX */

			_starpu_update_perfmodel_history(j, STARPU_GORDON_DEFAULT, cpuid, measured);
		}

		_starpu_push_task_output(j->task, 0);
		_starpu_handle_job_termination(j);
		//starpu_wake_all_blocked_workers();

		task_cnt++;
	}

	/* the job list was allocated by the gordon driver itself */
	starpu_job_list_delete(wrapper_list);

	starpu_wake_all_blocked_workers();
	free(task_wrapper->gordon_job);
	free(task_wrapper);
}


static void gordon_callback_func(void *arg)
{
	struct gordon_task_wrapper_s *task_wrapper = arg; 

	/* we don't know who will execute that codelet : so we actually defer the
 	 * execution of the StarPU codelet and the job termination later */
	struct starpu_worker_s *worker = task_wrapper->worker;
	STARPU_ASSERT(worker);

	task_wrapper->terminated = 1;

	task_wrapper->j->task->cl->per_worker_stats[worker->workerid]++;

	handle_terminated_job(task_wrapper->j);

	starpu_wake_all_blocked_workers();
	free(task_wrapper);
}

int inject_task(starpu_job_t j, struct starpu_worker_s *worker)
{
	struct starpu_task *task = j->task;
	int ret = _starpu_fetch_task_input(task, 0);

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

int inject_task_list(struct starpu_job_list_s *list, struct starpu_worker_s *worker)
{
	/* first put back all tasks that can not be performed by Gordon */
	unsigned nvalids = 0;
	unsigned ninvalids = 0;
	starpu_job_t j;

	// TODO !
//	
//	for (j = starpu_job_list_begin(list); j != starpu_job_list_end(list); j = starpu_job_list_next(j) )
//	{
//		if (!STARPU_GORDON_MAY_PERFORM(j)) {
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
	for (j = starpu_job_list_begin(list), index = 0; j != starpu_job_list_end(list); j = starpu_job_list_next(j), index++)
	{
		int ret;

		struct starpu_task *task = j->task;
		ret = _starpu_fetch_task_input(task, 0);
		STARPU_ASSERT(!ret);

		gordon_jobs[index].index = task->cl->gordon_func;

		struct starpu_perfmodel_t *model = j->task->cl->model;
		if (model && model->benchmarking)
			gordon_jobs[index].flags.sampling = 1;

		/* we should not hardcore the memory node ... XXX */
		unsigned memory_node = 0;
		starpu_to_gordon_buffers(j, &gordon_jobs[index], memory_node);
		
	}

	gordon_pushjob(task_wrapper->gordon_job, gordon_callback_list_func, task_wrapper);

	return 0;
}

void *gordon_worker_inject(struct starpu_worker_set_s *arg)
{

	while(_starpu_machine_is_running()) {
		if (gordon_busy_enough()) {
			/* gordon already has enough work, wait a little TODO */
			_starpu_wait_on_sched_event();
		}
		else {
#ifndef NOCHAIN
			int ret = 0;
#warning we should look into the local job list here !

			struct starpu_job_list_s *list = _starpu_pop_every_task(STARPU_GORDON);
			/* XXX 0 is hardcoded */
			if (list)
			{
				/* partition lists */
				unsigned size = job_list_size(list);
				unsigned nchunks = (size<2*arg->nworkers)?size:(2*arg->nworkers);
				//unsigned nchunks = (size<arg->nworkers)?size:(arg->nworkers);

				/* last element may be a little smaller (by 1) */
				unsigned chunksize = size/nchunks;

				unsigned chunk;
				for (chunk = 0; chunk < nchunks; chunk++)
				{
					struct starpu_job_list_s *chunk_list;
					if (chunk != (nchunks -1))
					{
						/* split the list in 2 parts : list = chunk_list | tail */
						chunk_list = starpu_job_list_new();

						/* find the end */
						chunk_list->_head = list->_head;

						starpu_job_itor_t it_j = starpu_job_list_begin(list);
						unsigned ind;
						for (ind = 0; ind < chunksize; ind++)
						{
							it_j = starpu_job_list_next(it_j);
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
				_starpu_wait_on_sched_event();
			}
#else
			/* gordon should accept a little more work */
			starpu_job_t j;
			j =  _starpu_pop_task();
	//		fprintf(stderr, "pop task %p\n", j);
			if (j) {
				if (STARPU_GORDON_MAY_PERFORM(j)) {
					/* inject that task */
					/* XXX we hardcore &arg->workers[0] for now */
					inject_task(j, &arg->workers[0]);
				}
				else {
					_starpu_push_task(j);
				}
			}
#endif
			
		}
	}

	return NULL;
}

void *_starpu_gordon_worker(void *arg)
{
	struct starpu_worker_set_s *gordon_set_arg = arg;

	_starpu_bind_thread_on_cpu(gordon_set_arg->config, gordon_set_arg->workers[0].bindid);

	/* TODO set_local_memory_node per SPU */
	gordon_init(gordon_set_arg->nworkers);	

	/* NB: On SPUs, the worker_key is set to NULL since there is no point
	 * in associating the PPU thread with a specific SPU (worker) while
	 * it's handling multiple processing units. */
	_starpu_set_local_worker_key(NULL);

	/* TODO set workers' name field */
	unsigned spu;
	for (spu = 0; spu < gordon_set_arg->nworkers; spu++)
	{
		struct starpu_worker_s *worker = &gordon_set_arg->workers[spu];
		snprintf(worker->name, 32, "SPU %d", worker->id);
	}

	/*
 	 * To take advantage of PPE being hyperthreaded, we should have 2 threads
 	 * for the gordon driver : one injects works, the other makes sure that
 	 * gordon is progressing (and performs the callbacks).
	 */

	/* launch the progression thread */
	pthread_mutex_init(&progress_mutex, NULL);
	pthread_cond_init(&progress_cond, NULL);
	
	pthread_create(&progress_thread, NULL, gordon_worker_progress, gordon_set_arg);

	/* wait for the progression thread to be ready */
	PTHREAD_MUTEX_LOCK(&progress_mutex);
	if (!progress_thread_is_inited)
		pthread_cond_wait(&progress_cond, &progress_mutex);
	PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	fprintf(stderr, "progress thread is running ... \n");
	
	/* tell the core that gordon is ready */
	PTHREAD_MUTEX_LOCK(&gordon_set_arg->mutex);
	gordon_set_arg->set_is_initialized = 1;
	pthread_cond_signal(&gordon_set_arg->ready_cond);
	PTHREAD_MUTEX_UNLOCK(&gordon_set_arg->mutex);

	gordon_worker_inject(gordon_set_arg);

	fprintf(stderr, "gordon deinit...\n");
	gordon_deinit();
	fprintf(stderr, "gordon was deinited\n");

	pthread_exit((void *)0x42);
}
