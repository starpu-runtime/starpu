/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_config.h>
#include <starpu_scheduler.h>
#include <schedulers/starpu_scheduler_toolbox.h>
#include <schedulers/starpu_hplookahead.h>
#include <common/graph.h>
#include <core/task.h>
#include <sched_policies/fifo_queues.h>
#include <limits.h>
#include <core/workers.h>
#include <datawizard/memory_nodes.h>

#ifdef HAVE_AYUDAME_H
#include <Ayudame.h>
#endif

#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

#define TCPU 2
#define TGPU 8
#define MAXACTIVEWORKERS 30
#define MAXTASKSINWORKERQUEUE 200
#define STARPU_MAX_PIPELINE 4

extern int _starpu_graph_record;

/* Each type of task will have a different queue, sorted in decreaing order of CPU affinity */
struct _starpu_hplookahead_task_queue
{
	/* codelet would be unique id for each type of tasks*/
	struct starpu_codelet *task_codelet;
	/* queue for each type of tasks*/
	struct starpu_st_fifo_taskq* tasks_queue;
	/* accelration factor for tasks of this queue (CPU time / GPU time)*/
	double acceleration_factor;
};

/* To store the tasks which become ready during simulation*/
struct _starpu_hplookahead_virtual_task_queues_table
{
	struct starpu_codelet *task_codelet;
	struct starpu_task*  tasks_queue[STARPU_HPLOOKAHEAD_NTASKSPERQUEUEINSIMULATION];
	/* both indices should be set to 0 before simulation starts */
	int start_index;
	int close_index;
};

struct _starpu_hplookahead_ready_queue
{
	unsigned  types_of_tasks;
	unsigned  ntasks;
	struct _starpu_hplookahead_task_queue task_queues[STARPU_HPLOOKAHEAD_MAXTYPESOFTASKS];
//    starpu_pthread_mutex_t hp_mutex;
};

struct _starpu_hp_lookahead_data
{
	/* data structure for ready queues*/
	struct _starpu_hplookahead_ready_queue *ready_queues;
	/* Allocated ready tasks to each worker*/
	struct starpu_st_fifo_taskq **workers_queue;

	struct _starpu_hplookahead_virtual_task_queues_table virtual_ready_queues[STARPU_HPLOOKAHEAD_MAXTYPESOFTASKS];
	unsigned types_of_tasks_in_simulation;

	/* one CPU worker to perform scheduling */
	unsigned schedulerWorker;

	/*an array variable holds the expected start time on each worker */
	double *exp_start_in_simulation;
	struct starpu_task **presentTaskInSimulation;

	struct starpu_task ***presentTaskInExecution;
	unsigned *presentIndexOfTaskInExecution;
	unsigned *isTaskInExecution;

	unsigned nGPUs;

	/* Variable to store number of active workers */
	unsigned nWorkers;
	/* local indices to worker id mapping */
	unsigned int *localIndicesToWorkerId;
	unsigned *nEntriesInWQs, *nExaminedEntriesInWQs;
	/* A table to store tasks of worker queues*/
	/* TODO: allocate at runtime in add worker function
	 * Also verify the second dimension of matrix*/
	struct starpu_task *workersQueue[MAXACTIVEWORKERS][MAXTASKSINWORKERQUEUE];

	/* mutex variable to take a big lock on scheduler */
	starpu_pthread_mutex_t hp_mutex;
	/*mutex variable to protect ready queues */
	starpu_pthread_mutex_t ready_queues_mutex;
};

static inline unsigned getVirtualIndex(unsigned index, unsigned *localIndicesToWorkerId, unsigned nWorkers)
{
	unsigned i;
	for(i=0; i<nWorkers; i++)
	{
		if(localIndicesToWorkerId[i] == index)
			return i;
	}
	STARPU_ABORT_MSG("First time seeing worker with %u id\n", index);
	return 0;
}

#if 0
static void displayQueues(struct _starpu_hp_lookahead_data *data)
{
	unsigned i;
	if(data->ready_queues->types_of_tasks > 0)
	{
		_STARPU_DEBUG("***** Ready Queues*******\n");

		for(i=0; i < data->ready_queues->types_of_tasks; i++)
		{
			_STARPU_DEBUG("Ready Queue %d:", i);
			struct starpu_task *task = starpu_task_list_front(&data->ready_queues->task_queues[i].tasks_queue->taskq);
			while(task != NULL)
			{
				_STARPU_DEBUG(" %lu", task->tag_id);
				task = task->next;
			}
			_STARPU_DEBUG("\n");
		}

		_STARPU_DEBUG("***** Worker Queues*******\n");
		for(i=0; i<data->nWorkers; i++)
		{
			unsigned workerId = data->localIndicesToWorkerId[i];
			struct starpu_task *task = starpu_task_list_front(&data->workers_queue[workerId]->taskq);
			_STARPU_DEBUG("Worker Queue %d (expStart=%lf, expLength=%lf):", workerId, data->workers_queue[workerId]->exp_start, data->workers_queue[workerId]->exp_len);
			while(task)
			{
				_STARPU_DEBUG(" %lu", task->tag_id);
				task = task->next;
			}
			_STARPU_DEBUG("\n");
		}
	}
}
#endif

#if 0
static void displayVT(struct _starpu_hp_lookahead_data *data)
{
	unsigned i;
	for(i=0; i<data->types_of_tasks_in_simulation; i++)
	{
		int j;
		_STARPU_DEBUG("VQ %d(startIndex=%d,endIndex=%d):", i, data->virtual_ready_queues[i].start_index, data->virtual_ready_queues[i].close_index);
		for(j=data->virtual_ready_queues[i].start_index; j<data->virtual_ready_queues[i].close_index; j++)
		{
			_STARPU_DEBUG(" %lu", data->virtual_ready_queues[i].tasks_queue[j]->tag_id);
		}
		_STARPU_DEBUG("\n");
	}
}
#endif

static inline void addToVirtualqueue(struct starpu_task *task, struct _starpu_hp_lookahead_data *data)
{
	unsigned i;
	for(i=0; i<data->types_of_tasks_in_simulation; i++)
	{
		if(data->virtual_ready_queues[i].task_codelet == task->cl)
		{
			int endIndex = data->virtual_ready_queues[i].close_index;
			data->virtual_ready_queues[i].tasks_queue[endIndex] = task;
			data->virtual_ready_queues[i].close_index++;
			return;
		}
	}

	/* A new ttype of task has been found in simulation -- make a separate entry for this task*/
	data->virtual_ready_queues[i].task_codelet = task->cl;
	data->virtual_ready_queues[i].tasks_queue[0] = task;
	data->virtual_ready_queues[i].start_index = 0;
	data->virtual_ready_queues[i].close_index = 1;
	data->types_of_tasks_in_simulation++;
	return;
}

static void releaseAndAddtoVT(struct starpu_task *task, struct _starpu_hp_lookahead_data *data, struct starpu_task *taskInConsideration)
{
	(void)task;
	(void)data;
	(void)taskInConsideration;
//    struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
//    unsigned int i;
//    for(i=0; i<j->n_outgoing; i++)
//    {
//        struct _starpu_job *jchild = j->outgoing[i];
//
//        if(!jchild)
//            continue;
//
//        unsigned l;
//        unsigned alreadyVisited = 0;
//        for(l=0; l<i; l++)
//        {
//            if(j->outgoing[l] == jchild)
//            {
//                alreadyVisited = 1;
//                break;
//            }
//        }
//        if(alreadyVisited == 1)
//            continue;
//
//        if(jchild->presentTaskInSimulation != taskInConsideration)
//        {
//            /* TODO: need to verify with Sam*/
//            /* Number of remaining incoming edges */
//            unsigned k;
//            unsigned remainingEdges = 0;
//            for(k=0; k<jchild->n_incoming; k++)
//            {
//                if(!jchild->incoming[k])
//                    continue;
//                remainingEdges ++;
//            }
//            jchild->ndeps_remaining_in_simulation = remainingEdges;
//        }
//        /*One task may create more than one dependency to another task*/
//        unsigned k;
//        for(k=0; k<jchild->n_incoming; k++)
//            if(jchild->incoming[k] == j)
//                jchild->ndeps_remaining_in_simulation--;
//
//        if(jchild->ndeps_remaining_in_simulation == 0)
//        {
//            if(jchild->task->cl != NULL)
//                addToVirtualqueue(jchild->task, data);
//            else
//                releaseAndAddtoVT(jchild->task, data, taskInConsideration);
//        }
//    }
}

static void init_hp_lookahead_sched(unsigned sched_ctx_id)
{
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)malloc(sizeof(struct _starpu_hp_lookahead_data));
	data->ready_queues = (struct _starpu_hplookahead_ready_queue *)malloc(sizeof(struct _starpu_hplookahead_ready_queue));
	data->ready_queues->types_of_tasks = 0;
	data->ready_queues->ntasks = 0;

	data->workers_queue = (struct starpu_st_fifo_taskq**)malloc(STARPU_NMAXWORKERS*sizeof(struct starpu_st_fifo_taskq*));

	data->exp_start_in_simulation = (double *)malloc(STARPU_NMAXWORKERS*sizeof(double));
	data->presentTaskInSimulation= (struct starpu_task **)malloc(STARPU_NMAXWORKERS*sizeof(struct starpu_task *));
	data->presentTaskInExecution = (struct starpu_task ***)malloc(STARPU_NMAXWORKERS*sizeof(struct starpu_task **));
	data->presentIndexOfTaskInExecution = (unsigned *)malloc(STARPU_NMAXWORKERS * sizeof(unsigned));
	data->isTaskInExecution = (unsigned *)malloc(STARPU_NMAXWORKERS * sizeof(unsigned));
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		data->workers_queue[i] = NULL;
		data->presentTaskInExecution[i] = NULL;
		data->isTaskInExecution[i] = 0;
	}
	/* Create a condition variable to protect it */

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);

	starpu_pthread_mutex_init(&data->hp_mutex, NULL);
	starpu_pthread_mutex_init(&data->ready_queues_mutex, NULL);

	/* record graph structure of jobs in terms of incoming/outgoing edges */
	_starpu_graph_record = 1;
	_STARPU_DEBUG("Initialising hp scheduler\n");
}

static void deinit_hp_lookahead_sched(unsigned sched_ctx_id)
{
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned queueId;
	for(queueId=0; queueId <data->ready_queues->types_of_tasks; queueId++)
	{
		STARPU_ASSERT(starpu_st_fifo_taskq_empty(data->ready_queues->task_queues[queueId].tasks_queue) != 0);
		starpu_st_fifo_taskq_destroy(data->ready_queues->task_queues[queueId].tasks_queue);
	}

	starpu_pthread_mutex_destroy(&data->hp_mutex);
	starpu_pthread_mutex_destroy(&data->ready_queues_mutex);
	free(data->ready_queues);
	free(data->workers_queue);
	free(data->exp_start_in_simulation);
	free(data->presentTaskInExecution);
	free(data->presentIndexOfTaskInExecution);
	free(data->presentTaskInSimulation);
	free(data->isTaskInExecution);
	free(data);

	_STARPU_DEBUG("Deinitializing hp-lookahead scheduler\n");
}

static void hp_lookahead_add_workers(unsigned sched_ctx_id, int *workerIds, unsigned nworkers)
{
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned nGPUs = 0;
	unsigned workerId;
	unsigned i;

	data->nWorkers = nworkers;
	data->localIndicesToWorkerId = (unsigned *)malloc(nworkers * sizeof(unsigned));
	data->nEntriesInWQs = (unsigned *)malloc(nworkers * sizeof(unsigned));
	data->nExaminedEntriesInWQs = (unsigned *)malloc(nworkers * sizeof(unsigned));

	for (i = 0; i < nworkers; i++)
	{
		workerId = workerIds[i];
		/*TODO: Assumption: order of workers do not change*/
		data->localIndicesToWorkerId[i] = workerId;
		/* if the worker has alreadry belonged to this context
		   the queue and the synchronization variables have been already initialized */
		if(data->workers_queue[workerId] == NULL)
		{
			data->workers_queue[workerId] = starpu_st_fifo_taskq_create();
			data->workers_queue[workerId]->exp_start = 0;
			data->workers_queue[workerId]->exp_len = 0;
		}
		/* Allocate memory to store all tasks in execution to fulfill pipeline length*/
		data->presentTaskInExecution[workerId] = (struct starpu_task **)malloc(STARPU_MAX_PIPELINE * sizeof(struct starpu_task *));
		unsigned j;
		for(j=0; j<STARPU_MAX_PIPELINE; j++)
			data->presentTaskInExecution[workerId][j] = NULL;

		data->presentIndexOfTaskInExecution[workerId] = 0;
		if(starpu_worker_get_type(workerId) == STARPU_CUDA_WORKER)
			nGPUs++;
	}
	data->nGPUs = nGPUs;

	for (i = 0; i < nworkers; i++)
	{
		workerId = workerIds[i];
		if(starpu_worker_get_type(workerId) == STARPU_CPU_WORKER)
		{
			/* first CPU worker is doing allocation for all other workers*/
			data->schedulerWorker = workerId;
			return;
		}
	}
	STARPU_ABORT_MSG("Present Architecture does not have any CPU\n");
}

static void hp_lookahead_remove_workers(unsigned sched_ctx_id, int *workerIds, unsigned nworkers)
{
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned workerId;
	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		workerId = workerIds[i];
		if(data->workers_queue[workerId] != NULL)
		{
			starpu_st_fifo_taskq_destroy(data->workers_queue[workerId]);
			data->workers_queue[workerId] = NULL;
		}
		free(data->presentTaskInExecution[workerId]);
	}
	free(data->localIndicesToWorkerId);
	free(data->nEntriesInWQs);
	free(data->nExaminedEntriesInWQs);
}

static double compute_acceleration_ratio(struct starpu_task *task, unsigned sched_ctx_id)
{
	/******* Calculation of max(CPU time) / min(GPU time) ******/
	double cpu_time = DBL_MIN;
	double gpu_time = DBL_MAX;
	unsigned impl_mask;
	unsigned worker, nimpl;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	/*TODO: verify with Sam :difference between init_iterator and init_iterator_for_parallel_tasks */
	/*Any optimized method (through array entries or something similar) to iterate over active worker list */
	struct starpu_sched_ctx_iterator it;
	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(worker, sched_ctx_id);
		if (!starpu_worker_can_execute_task_impl(worker, task, &impl_mask))
			continue;

		if(starpu_worker_get_type(worker) == STARPU_CPU_WORKER)
		{
			for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (!(impl_mask & (1U << nimpl)))
				{
					/* no one on that queue may execute this task */
					continue;
				}
				double expected_length = starpu_task_expected_length(task, perf_arch, nimpl);
				if(expected_length > cpu_time)
					cpu_time = expected_length;
			}
		}
		else if(starpu_worker_get_type(worker) == STARPU_CUDA_WORKER)
		{
			for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (!(impl_mask & (1U << nimpl)))
				{
					/* no one on that queue may execute this task */
					continue;
				}
				double expected_length = starpu_task_expected_length(task, perf_arch, nimpl);
				if(expected_length < gpu_time)
					gpu_time = expected_length;
			}
		}
		else
			/* other than CPU/GPU worker */
			continue;
	}
	/*To avoid division by zero in some undesirable situation
	 * add 1 to denominator
	 *
	 */
	return cpu_time/(gpu_time + 1.0);
}

static int  push_task_in_to_hp_lookahead_ready_queue(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	//Consider separately when a task doest not have codelets for both CPU & GPU resources
	if(!((task->cl->where & STARPU_CPU) && (task->cl->where & STARPU_CUDA)))
	{
		/* Task can not execute on both resources, CPU and GPU
		 * schedule task on the resource which is having less number
		 * of scheduled tasks in its worker queue
		 *
		 * */
		int chosen_worker = -1;
		unsigned chosen_impl = -1;
		unsigned worker;
		unsigned min_ntasks;
		unsigned impl_mask;
		starpu_pthread_mutex_lock(&data->ready_queues_mutex);
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

		struct starpu_sched_ctx_iterator it;
		workers->init_iterator_for_parallel_tasks(workers, &it, task);
		/* Find the worker which has less number of assigned ready tasks in its queue */
		while(workers->has_next(workers, &it))
		{
			worker = workers->get_next(workers, &it);
			/* Selct first implemenattion on the worker which can execute this task*/
			if (!starpu_worker_can_execute_task_first_impl(worker, task, &impl_mask))
				continue;
			if(chosen_worker == -1)
			{
				chosen_worker = worker;
				min_ntasks = data->workers_queue[worker]->ntasks;
				chosen_impl = impl_mask;
			}
			else if(min_ntasks > data->workers_queue[worker]->ntasks)
			{
				chosen_worker = worker;
				min_ntasks = data->workers_queue[worker]->ntasks;
				chosen_impl = impl_mask;
			}
		}
		STARPU_ASSERT_MSG(chosen_worker != -1, "Not any worker is able to execute some tasks");
		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(chosen_worker, sched_ctx_id);

		/* Expected duration of task on chosen worker */
		double model = starpu_task_expected_length(task, perf_arch, chosen_impl);
		task->predicted = model;

		unsigned memory_node = starpu_worker_get_memory_node(chosen_worker);
		starpu_prefetch_task_input_on_node(task, memory_node);

		struct starpu_st_fifo_taskq *fifo = data->workers_queue[chosen_worker];
		starpu_pthread_mutex_t *sched_mutex;
		starpu_pthread_cond_t *sched_cond;
		starpu_worker_get_sched_condition(chosen_worker, &sched_mutex, &sched_cond);
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
		fifo->exp_len  += model;
		starpu_st_fifo_taskq_push_task(fifo, task);
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
		starpu_push_task_end(task);
		starpu_pthread_mutex_unlock(&data->ready_queues_mutex);
		return 0;
	}
	/*Already locked inside job termination */
	//starpu_pthread_mutex_lock(&data->hp_mutex);

	starpu_pthread_mutex_lock(&data->ready_queues_mutex);
	/* Insert task in the appropriate queue */
	unsigned int queueId;
	for(queueId=0; queueId<data->ready_queues->types_of_tasks; queueId++)
	{
		/* checking whether other tasks with the same codelt have been
		 * recorded in the past or not */
		/************** pointer comparison **********/
		if(data->ready_queues->task_queues[queueId].task_codelet == task->cl)
		{
			//TODO: add tasks in nonincreaing order of priorities
			starpu_st_fifo_taskq_push_task(data->ready_queues->task_queues[queueId].tasks_queue, task);
			break;
		}
	}
	if(queueId == data->ready_queues->types_of_tasks)
	{
		//first instance of this type of task
		/* Create a new ready queue to store this type of tasks */
		struct starpu_st_fifo_taskq *current_taskq = starpu_st_fifo_taskq_create();
		starpu_st_fifo_taskq_push_task(current_taskq, task);
		//compute acceleration factor for this type of tasks
		double current_acceleration_factor = compute_acceleration_ratio(task, sched_ctx_id);
		/* insert the queues based on the nondecreasing order of acceleration factor */
		/* Insertion sort based on acceleration factor */
		while((queueId > 0) && (data->ready_queues->task_queues[queueId-1].acceleration_factor > current_acceleration_factor))
		{
			data->ready_queues->task_queues[queueId].task_codelet = data->ready_queues->task_queues[queueId - 1].task_codelet;
			data->ready_queues->task_queues[queueId].tasks_queue = data->ready_queues->task_queues[queueId - 1].tasks_queue;
			data->ready_queues->task_queues[queueId].acceleration_factor = data->ready_queues->task_queues[queueId - 1].acceleration_factor;
			queueId--;
		}
		/* pointer to task codelet */
		data->ready_queues->task_queues[queueId].task_codelet = task->cl;
		data->ready_queues->task_queues[queueId].tasks_queue = current_taskq;
		data->ready_queues->task_queues[queueId].acceleration_factor = current_acceleration_factor;

		data->ready_queues->types_of_tasks ++;
	}

	data->ready_queues->ntasks ++;

	starpu_push_task_end(task);
	/*Already locked inside post exec */
	//starpu_pthread_mutex_unlock(&data->hp_mutex);

	starpu_pthread_mutex_unlock(&data->ready_queues_mutex);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	/*if there are no tasks block */
	/* wake people waiting for a task */
	unsigned worker = 0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		starpu_pthread_mutex_t *sched_mutex;
		starpu_pthread_cond_t *sched_cond;
		starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
		starpu_pthread_mutex_lock(sched_mutex);
		starpu_pthread_cond_signal(sched_cond);
		starpu_pthread_mutex_unlock(sched_mutex);
	}
#endif
	return 0;
}

static struct starpu_task *pop_task_from_hp_lookahead_ready_queue(unsigned sched_ctx_id)
{
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned workerId = starpu_worker_get_id();

	/* A big lock : may impact the performance, but required for advance lookup*/

	if(workerId == data->schedulerWorker)
	{
		starpu_pthread_mutex_lock(&data->hp_mutex);
		starpu_pthread_mutex_lock(&data->ready_queues_mutex);

		/* If ready queue is empty then nothing to schedule : simply return*/
		if(data->ready_queues->ntasks == 0)
			goto afterAnalyzing;

		int ntaskInTheBeginning = data->ready_queues->ntasks;
		unsigned i;
		double minBusyTime = DBL_MAX;

		/* potential worker id on which next task can be allocated */
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
		struct starpu_sched_ctx_iterator it;
		unsigned worker;
		workers->init_iterator(workers, &it);
		while(workers->has_next(workers, &it))
		{
			worker = workers->get_next(workers, &it);
			if(worker == data->schedulerWorker)
				continue;

			struct starpu_st_fifo_taskq *fifo = data->workers_queue[worker];
			fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
			double completionTimeOnThisWorker = fifo->exp_start + fifo->exp_len;
			if(completionTimeOnThisWorker < minBusyTime)
			{
				minBusyTime = completionTimeOnThisWorker;
				workerId = worker;
			}
			else if((completionTimeOnThisWorker == minBusyTime) &&(starpu_worker_get_type(worker) == STARPU_CUDA_WORKER))
			{
				minBusyTime = completionTimeOnThisWorker;
				workerId = worker;
			}
		}

		struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerId];
		if((starpu_worker_get_type(workerId) == STARPU_CPU_WORKER) && (fifo->ntasks <= TCPU))
		{
			/* present worker is a CPU worker therefore it performs
			 * simulation to determine the allocation of task which is best suited fo this worker*/
			/*Try to schedule a task on CPU worker whose id is workerId */
			struct starpu_task *taskInConsideration = NULL;
			unsigned queueId;
			/* Mark the highest priority task from the less accelerated queue */
			/* Decison of this task is used tp terminate the whole simulation process */
			for(queueId=0; queueId < data->ready_queues->types_of_tasks && taskInConsideration == NULL; queueId++)
				taskInConsideration = starpu_task_list_front(&data->ready_queues->task_queues[queueId].tasks_queue->taskq);

			/* Store expected start time of each worker and present task in execution into temporary buffer */
			unsigned worker;
			for(worker =0; worker < data->nWorkers; worker++)
			{
				unsigned originalWorker = data->localIndicesToWorkerId[worker];
				if(originalWorker == data->schedulerWorker)
					continue;
				/* No worker has exp_start time less than current time */
				data->workers_queue[originalWorker]->exp_start = STARPU_MAX(data->workers_queue[originalWorker]->exp_start, starpu_timing_now());
				data->exp_start_in_simulation[worker] = data->workers_queue[originalWorker]->exp_start;
				unsigned index = data->presentIndexOfTaskInExecution[originalWorker];

				struct starpu_task *task = data->presentTaskInExecution[originalWorker][index];
				unsigned j=0;
				if(task == NULL)
				{
					j = (index+1)% STARPU_MAX_PIPELINE;
					for(; task != NULL &&  j!=index; j = (j+1)%STARPU_MAX_PIPELINE)
						task = data->presentTaskInExecution[originalWorker][j];
				}

				data->presentTaskInSimulation[worker] = task;
				/* If Worker did not yet start the next task of pipeline buffer */
				if((data->isTaskInExecution[originalWorker] == 0) && task != NULL)
					data->exp_start_in_simulation[worker] += task->predicted;

				unsigned counter = 0;
				if(task != NULL)
				{
					j = (j+1)%STARPU_MAX_PIPELINE;

					while(j != index)
					{
						task = data->presentTaskInExecution[originalWorker][j];
						STARPU_ASSERT(data->presentTaskInExecution[originalWorker][j] != NULL);
						data->workersQueue[worker][counter] = task;
						j = (j+1)%STARPU_MAX_PIPELINE;
						counter++;
					}
				}
				data->nEntriesInWQs[worker] = counter;
			}

			/*Copy the codelet information from ready queues to virtual table */
			/*Optimization: It can be assigned when a new entry is created in ready queues */
			data->types_of_tasks_in_simulation = data->ready_queues->types_of_tasks;
			for(i=0; i<data->types_of_tasks_in_simulation; i++)
			{
				data->virtual_ready_queues[i].task_codelet = data->ready_queues->task_queues[i].task_codelet;
				data->virtual_ready_queues[i].start_index = 0;
				data->virtual_ready_queues[i].close_index = 0;
			}
			/*copy worker queues to table structure */
			/* possible to replace this table with the present set of tasks (an array) in consideration*/
			unsigned nWorkers = data->nWorkers;
			unsigned virtualWorkerId;
			for(virtualWorkerId=0; virtualWorkerId<nWorkers; virtualWorkerId++)
			{
				unsigned workerId = data->localIndicesToWorkerId[virtualWorkerId];

				if(workerId == data->schedulerWorker)
					continue;

				struct starpu_task *task = starpu_task_list_front(&data->workers_queue[workerId]->taskq);
				unsigned counter = data->nEntriesInWQs[virtualWorkerId];
				while(task)
				{
					data->workersQueue[virtualWorkerId][counter] = task;
					task = task->next;
					counter++;
				}
				data->nEntriesInWQs[virtualWorkerId] = counter;
				data->nExaminedEntriesInWQs[virtualWorkerId] = 0;
			}
			/* these two set of variables improve the scan process of ready queues */
			int cpuReadyQueueIndex = 0;
			struct starpu_task *cpuReadyTask = NULL;

			int gpuReadyQueueIndex = data->ready_queues->types_of_tasks - 1;
			struct starpu_task *gpuReadyTask = NULL;

			/* All temporary structures are set */
			/* Start the simulation until final decision for task taskInConsideration is taken */
		repeatWhileTrue:
			while(1)
			{
				/* Find the worker who is going to be idle soon */
				/* In case of tie between CPU and GPU, GPU would be preferred) */
				double minStartTime =  DBL_MAX;
				unsigned minIndex=0;
				unsigned worker;
				for(worker=0; worker < nWorkers; worker++)
				{
					unsigned originalWorker = data->localIndicesToWorkerId[worker];

					if(originalWorker == data->schedulerWorker)
						continue;

					if(data->exp_start_in_simulation[worker] < minStartTime)
					{
						minStartTime = data->exp_start_in_simulation[worker];
						minIndex = worker;
					}
					else if((data->exp_start_in_simulation[worker] == minStartTime) && (starpu_worker_get_type(originalWorker) == STARPU_CUDA_WORKER))
					{
						minIndex = worker;
					}
				}

				/* release all tasks finished at minStartTime */
				for(worker=0; worker < nWorkers; worker++)
				{
					if((data->exp_start_in_simulation[worker] == minStartTime) && (data->presentTaskInSimulation[worker] != NULL))
					{
						if(data->presentTaskInSimulation[worker] == taskInConsideration)
						{
							STARPU_ASSERT_MSG(ntaskInTheBeginning == data->ready_queues->ntasks , "Number of ready tasks before starting the simulation was %d but now it changed to %d\n", ntaskInTheBeginning, data->ready_queues->ntasks);
							/*taskInConsideration complete its execution on CPU */
							/* It is safe to execute this task on a CPU worker whose id is worker */
							/* Optimization: May consider allocation of  all tasks (ready tasks , non ready tasks) which has been considered while taking decision for this task */

							STARPU_ASSERT_MSG(ntaskInTheBeginning == data->ready_queues->ntasks , "At line number %d, Number of ready tasks before starting the simulation was %d but now it changed to %d\n",  __LINE__, ntaskInTheBeginning, data->ready_queues->ntasks);
							struct starpu_task *task = NULL;
							int queueId;
							for(queueId = 0; queueId < data -> ready_queues->types_of_tasks && task == NULL; queueId++)
								task = starpu_st_fifo_taskq_pop_local_task(data->ready_queues->task_queues[queueId].tasks_queue);

							STARPU_ASSERT_MSG(task == taskInConsideration, "At line number %d, next task is different from the task for which simulation has started", __LINE__);
							unsigned originalWorker = data->localIndicesToWorkerId[worker];
							struct starpu_st_fifo_taskq *fifo = data->workers_queue[originalWorker];

							struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalWorker, sched_ctx_id);

							/* Expected duration of task on current worker */
							double model = starpu_task_expected_length(task, perf_arch, 0);
							task->predicted = model;

							unsigned memory_node = starpu_worker_get_memory_node(originalWorker);
							starpu_prefetch_task_input_on_node(task, memory_node);

							fifo->exp_len  += model;
							starpu_st_fifo_taskq_push_task(fifo, task);
							data->ready_queues->ntasks --;
							// Terminate the simulation
							goto afterAnalyzing;
						}
						else
						{
							/* releae tasks which become ready after completion of this tasks and add to virtual simulation table */
							releaseAndAddtoVT(data->presentTaskInSimulation[worker], data, taskInConsideration);
							data->presentTaskInSimulation[worker] = NULL;
						}
					}
				}

				/* Select a task from workers queue */
				if(data->nExaminedEntriesInWQs[minIndex] < data->nEntriesInWQs[minIndex])
				{
					struct starpu_task *task = data->workersQueue[minIndex][data->nExaminedEntriesInWQs[minIndex]];
					data->nExaminedEntriesInWQs[minIndex]++;
					unsigned workerId = data->localIndicesToWorkerId[minIndex];
					struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerId, sched_ctx_id);
					/* Expected duration of task on current worker */
					double model = starpu_task_expected_length(task, perf_arch, 0);
					data->exp_start_in_simulation[minIndex] += model;
					data->presentTaskInSimulation[minIndex] = task;

					unsigned AminIndex = data->localIndicesToWorkerId[minIndex];

					goto repeatWhileTrue;
				}

				/* set the execution of task  present on the worker with minIndex Id */
				data->presentTaskInSimulation[minIndex] = NULL;
				unsigned originalMinIndex = data->localIndicesToWorkerId[minIndex];
				if(starpu_worker_get_type(originalMinIndex) == STARPU_CPU_WORKER)
				{
					{
						/*try to take a task from virtualQueueTable */
						int i;
						for(i=0; i<cpuReadyQueueIndex; i++)
						{
							int startIndex = data->virtual_ready_queues[i].start_index;
							int closeIndex = data->virtual_ready_queues[i].close_index;
							if(startIndex < closeIndex)
							{
								struct starpu_task *task =data->virtual_ready_queues[i].tasks_queue[startIndex];
								data->virtual_ready_queues[i].start_index ++;
								struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
								/* Expected duration of task on current worker */
								double model = starpu_task_expected_length(task, perf_arch, 0);
								data->exp_start_in_simulation[minIndex] += model;
								data->presentTaskInSimulation[minIndex] = task;

								unsigned AminIndex = data->localIndicesToWorkerId[minIndex];

								goto repeatWhileTrue;
							}
						}
					}
					{
						/* Try to take a task by analyzing both table as well as ready queues*/
						int i;
						for(i=cpuReadyQueueIndex; i <= gpuReadyQueueIndex; i++)
						{
							int startIndex = data->virtual_ready_queues[i].start_index;
							int closeIndex = data->virtual_ready_queues[i].close_index;
							if(((cpuReadyTask == NULL) && (data->ready_queues->task_queues[i].tasks_queue->ntasks > 0)) || ((cpuReadyTask  != NULL) && (cpuReadyTask->next != NULL)))
							{
								/* There is some task in the ith ready queue */
								struct starpu_task *firstTask;
								if(cpuReadyTask == NULL)
									firstTask = starpu_task_list_front(&data->ready_queues->task_queues[i].tasks_queue->taskq);
								else
									firstTask  = cpuReadyTask->next;
								if(startIndex != closeIndex)
								{
									/*Also there is some task in the virtaul table of ith queue*/
									struct starpu_task *secondTask = data->virtual_ready_queues[i].tasks_queue[startIndex];
									if(firstTask->priority >= secondTask->priority)
									{
										cpuReadyTask = firstTask;
										struct starpu_task *task = firstTask;
										struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
										/* Expected duration of task on current worker */
										double model = starpu_task_expected_length(task, perf_arch, 0);
										data->exp_start_in_simulation[minIndex] += model;
										data->presentTaskInSimulation[minIndex] = task;

										goto repeatWhileTrue;
									}
									else
									{
										data->virtual_ready_queues[i].start_index ++;
										struct starpu_task *task = secondTask;
										struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
										/* Expected duration of task on current worker */
										double model = starpu_task_expected_length(task, perf_arch, 0);
										data->exp_start_in_simulation[minIndex] += model;
										data->presentTaskInSimulation[minIndex] = task;

										goto repeatWhileTrue;
									}
								}
								else
								{
									cpuReadyTask = firstTask;
									struct starpu_task *task = firstTask;
									struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
									/* Expected duration of task on current worker */
									double model = starpu_task_expected_length(task, perf_arch, 0);
									data->exp_start_in_simulation[minIndex] += model;
									data->presentTaskInSimulation[minIndex] = task;

									goto repeatWhileTrue;
								}
							}
							else if(startIndex < closeIndex)
							{
								struct starpu_task *task =data->virtual_ready_queues[i].tasks_queue[startIndex];
								data->virtual_ready_queues[i].start_index ++;
								struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
								/* Expected duration of task on current worker */
								double model = starpu_task_expected_length(task, perf_arch, 0);
								data->exp_start_in_simulation[minIndex] += model;
								data->presentTaskInSimulation[minIndex] = task;

								goto repeatWhileTrue;
							}
							else
							{
								cpuReadyQueueIndex++;
								cpuReadyTask = NULL;
							}
						}
					}
					{
						/*try to take a task from virtualQueueTable */
						int i;
						for(i=gpuReadyQueueIndex + 1; i<data->types_of_tasks_in_simulation; i++)
						{
							int startIndex = data->virtual_ready_queues[i].start_index;
							int closeIndex = data->virtual_ready_queues[i].close_index;
							if(startIndex < closeIndex)
							{
								struct starpu_task *task =data->virtual_ready_queues[i].tasks_queue[startIndex];
								data->virtual_ready_queues[i].start_index ++;
								struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
								/* Expected duration of task on current worker */
								double model = starpu_task_expected_length(task, perf_arch, 0);
								data->exp_start_in_simulation[minIndex] += model;
								data->presentTaskInSimulation[minIndex] = task;

								goto repeatWhileTrue;
							}
						}
					}
					/* If control comes here: It indicates that there are not ready tasks to schedule at this moment */
					/* Find the next eventpoint and update the exp_start time for each worker appropriately*/
					{
						double nextEventPoint = DBL_MAX;
						unsigned worker;
						/* Proceed to next iteration find a new eventpoint */
						for(worker = 0; worker <data->nWorkers; worker++)
						{
							unsigned originalWorker = data->localIndicesToWorkerId[worker];

							if(originalWorker == data->schedulerWorker)
								continue;

							if((data->presentTaskInSimulation[worker] != NULL) && (data->exp_start_in_simulation[worker] < nextEventPoint))
							{
								nextEventPoint = data->exp_start_in_simulation[worker];
							}
						}

						/* set next eventpoint for all idle workers */
						for(worker = 0; worker <data->nWorkers; worker++)
						{
							unsigned originalWorker = data->localIndicesToWorkerId[worker];

							if(originalWorker == data->schedulerWorker)
								continue;

							if(data->exp_start_in_simulation[worker] < nextEventPoint)
								data->exp_start_in_simulation[worker] = nextEventPoint;
						}

						goto repeatWhileTrue;
					}
				}
				else if(starpu_worker_get_type(originalMinIndex) == STARPU_CUDA_WORKER)
				{
					{
						/*try to take a task from virtualQueueTable */
						int i;
						for(i = data->types_of_tasks_in_simulation-1; i > gpuReadyQueueIndex;  i--)
						{
							int startIndex = data->virtual_ready_queues[i].start_index;
							int closeIndex = data->virtual_ready_queues[i].close_index;
							if(startIndex < closeIndex)
							{
								struct starpu_task *task =data->virtual_ready_queues[i].tasks_queue[startIndex];
								data->virtual_ready_queues[i].start_index ++;
								struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
								/* Expected duration of task on current worker */
								double model = starpu_task_expected_length(task, perf_arch, 0);
								data->exp_start_in_simulation[minIndex] += model;
								data->presentTaskInSimulation[minIndex] = task;

								goto repeatWhileTrue;
							}
						}
					}
					{
						/* Try to take a task by analyzing both table as well as ready queues*/
						int i;
						for(i=gpuReadyQueueIndex; i >= cpuReadyQueueIndex; i--)
						{
							int startIndex = data->virtual_ready_queues[i].start_index;
							int closeIndex = data->virtual_ready_queues[i].close_index;
							if(((gpuReadyTask == NULL) && (data->ready_queues->task_queues[i].tasks_queue->ntasks > 0)) || ((gpuReadyTask  != NULL) && (gpuReadyTask->next != NULL)))
							{
								/* There is some task in the ith ready queue */
								struct starpu_task *firstTask;
								if(gpuReadyTask == NULL)
									firstTask = starpu_task_list_front(&data->ready_queues->task_queues[i].tasks_queue->taskq);
								else
									firstTask  = gpuReadyTask->next;
								if(startIndex != closeIndex)
								{
									/*Also there is some task in the virtaul table of ith queue*/
									struct starpu_task *secondTask = data->virtual_ready_queues[i].tasks_queue[startIndex];
									if(firstTask->priority >= secondTask->priority)
									{
										gpuReadyTask = firstTask;
										struct starpu_task *task = firstTask;
										struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
										/* Expected duration of task on current worker */
										double model = starpu_task_expected_length(task, perf_arch, 0);
										data->exp_start_in_simulation[minIndex] += model;
										data->presentTaskInSimulation[minIndex] = task;

										goto repeatWhileTrue;
									}
									else
									{
										data->virtual_ready_queues[i].start_index ++;
										struct starpu_task *task = secondTask;
										struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
										/* Expected duration of task on current worker */
										double model = starpu_task_expected_length(task, perf_arch, 0);
										data->exp_start_in_simulation[minIndex] += model;
										data->presentTaskInSimulation[minIndex] = task;

										goto repeatWhileTrue;
									}
								}
								else
								{
									gpuReadyTask = firstTask;
									struct starpu_task *task = firstTask;
									struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
									/* Expected duration of task on current worker */
									double model = starpu_task_expected_length(task, perf_arch, 0);
									data->exp_start_in_simulation[minIndex] += model;
									data->presentTaskInSimulation[minIndex] = task;

									goto repeatWhileTrue;
								}
							}
							else if(startIndex < closeIndex)
							{
								struct starpu_task *task =data->virtual_ready_queues[i].tasks_queue[startIndex];
								data->virtual_ready_queues[i].start_index ++;
								struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
								/* Expected duration of task on current worker */
								double model = starpu_task_expected_length(task, perf_arch, 0);
								data->exp_start_in_simulation[minIndex] += model;
								data->presentTaskInSimulation[minIndex] = task;

								goto repeatWhileTrue;
							}
							else
							{
								gpuReadyQueueIndex--;
								gpuReadyTask = NULL;
							}
						}
					}
					{
						/*try to take a task from virtualQueueTable */
						int i;
						for(i = cpuReadyQueueIndex-1; i >= 0; i--)
						{
							int startIndex = data->virtual_ready_queues[i].start_index;
							int closeIndex = data->virtual_ready_queues[i].close_index;
							if(startIndex < closeIndex)
							{
								struct starpu_task *task =data->virtual_ready_queues[i].tasks_queue[startIndex];
								data->virtual_ready_queues[i].start_index ++;
								struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
								/* Expected duration of task on current worker */
								double model = starpu_task_expected_length(task, perf_arch, 0);
								data->exp_start_in_simulation[minIndex] += model;
								data->presentTaskInSimulation[minIndex] = task;

								goto repeatWhileTrue;

							}
						}
					}

					{
						/* If control comes here, it indicates that some GPU is idle*/
						/* calculate expected end time of taskInConsideration task on minIndex and
						 * compare with its expected completion time on CPU worker  */
						int cpuWorkerId = -1;
						unsigned worker;
						for(worker=0; worker < data->nWorkers; worker++)
						{
							/* scheduler worker will have NULL in current Task(as well as in simulation) in execution */
							if(data->presentTaskInSimulation[worker] == taskInConsideration)
							{
								cpuWorkerId = worker;
								break;
							}
						}
						STARPU_ASSERT_MSG(cpuWorkerId != -1, "Task in consideration is not running on CPU: something strange");
						STARPU_ASSERT_MSG(ntaskInTheBeginning == data->ready_queues->ntasks , "At line number %d, Number of ready tasks before starting the simulation was %d but now it changed to %d\n",  __LINE__, ntaskInTheBeginning, data->ready_queues->ntasks);
						/* pop task from ready queue */
						struct starpu_task *task = NULL;
						int queueId;
						for(queueId = 0; queueId<data -> ready_queues->types_of_tasks && task == NULL; queueId++)
							task = starpu_st_fifo_taskq_pop_local_task(data->ready_queues->task_queues[queueId].tasks_queue);

						data->ready_queues->ntasks --;

						STARPU_ASSERT_MSG(task == taskInConsideration, "At line number %d, next task is different from the task for which simulation has started", __LINE__);
						unsigned originalMinIndex = data->localIndicesToWorkerId[minIndex];
						struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
						/* Expected duration of task on current worker */
						double model = starpu_task_expected_length(taskInConsideration, perf_arch, 0);
						if(data->exp_start_in_simulation[minIndex] + model < data->exp_start_in_simulation[cpuWorkerId])
						{
							/* Find GPU worker whose expected completition time with allocated tasks is minimum */
							double minCOmpletionTimeOnGPUs = DBL_MAX;
							unsigned worker;
							for(worker=0; worker < data->nWorkers; worker++)
							{
								unsigned originalWorkerId = data->localIndicesToWorkerId[worker];
								if(starpu_worker_get_type(originalWorkerId)== STARPU_CUDA_WORKER)
								{
									struct starpu_st_fifo_taskq *fifo = data->workers_queue[originalWorkerId];
									if(fifo->exp_start + fifo->exp_len < minCOmpletionTimeOnGPUs)
									{
										minCOmpletionTimeOnGPUs = fifo->exp_start + fifo->exp_len;
										originalMinIndex = originalWorkerId;
									}
								}
							}
							/*place task in the minIndex GPU queue */
							struct starpu_st_fifo_taskq *fifo = data->workers_queue[originalMinIndex];
							struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);

							/* Expected duration of task on current worker */
							double model = starpu_task_expected_length(taskInConsideration, perf_arch, 0);
							taskInConsideration->predicted = model;

							unsigned memory_node = starpu_worker_get_memory_node(originalMinIndex);
							starpu_prefetch_task_input_on_node(taskInConsideration, memory_node);

							fifo->exp_len  += model;
							starpu_st_fifo_taskq_push_task(fifo, taskInConsideration);
						}
						else
						{
							/* Place task in the cpu worker queue */
							/*One optimization could be schedule all ready tasks (even non ready tasks) involved in for decision of taskInConsideration*/
							unsigned originalCPUId = data->localIndicesToWorkerId[cpuWorkerId];
							struct starpu_st_fifo_taskq *fifo = data->workers_queue[originalCPUId];
							struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalCPUId, sched_ctx_id);
							/* Expected duration of task on current worker */
							double model = starpu_task_expected_length(taskInConsideration, perf_arch, 0);
							taskInConsideration->predicted = model;

							unsigned memory_node = starpu_worker_get_memory_node(originalCPUId);
							starpu_prefetch_task_input_on_node(taskInConsideration, memory_node);

							fifo->exp_len  += model;
							starpu_st_fifo_taskq_push_task(fifo, taskInConsideration);
						}
						goto afterAnalyzing;
					}
				}
				/* end of While true loop */
			}
		}
		else if((starpu_worker_get_type(workerId) == STARPU_CUDA_WORKER) && (fifo->ntasks <= TGPU))
		{
			/* A GPU worker pop fair number of tasks from ready queues and
			 * put in its worker queue before popping a task from its worker queue */
			int nGPUs = data->nGPUs;
			int nFairAllocations = (data->ready_queues->ntasks + nGPUs-1)/nGPUs;
			while((nFairAllocations > 0) && (fifo->ntasks <= TGPU))
			{
				// while conditions ensure that GPU gets a nonnull task
				int queueId;
				struct starpu_task *task = NULL;
				for(queueId=data->ready_queues->types_of_tasks - 1; queueId >=0 && task == NULL; queueId--)
					task = starpu_st_fifo_taskq_pop_local_task(data->ready_queues->task_queues[queueId].tasks_queue);

				data->ready_queues->ntasks --;

				struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerId, sched_ctx_id);

				/* Expected duration of task on current worker */
				double model = starpu_task_expected_length(task, perf_arch, 0);
				task->predicted = model;

				unsigned memory_node = starpu_worker_get_memory_node(workerId);
				starpu_prefetch_task_input_on_node(task, memory_node);

				/*
				  starpu_pthread_mutex_t *sched_mutex;
				  starpu_pthread_cond_t *sched_cond;
				  starpu_worker_get_sched_condition(workerId, &sched_mutex, &sched_cond);
				  STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
				*/
				fifo->exp_len  += model;
				starpu_st_fifo_taskq_push_task(fifo, task);
				/*
				  STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
				*/
				nFairAllocations--;
			}
		}
	afterAnalyzing:
		starpu_pthread_mutex_unlock(&data->ready_queues_mutex);
		starpu_pthread_mutex_unlock(&data->hp_mutex);
		return NULL;
	}
	else if((starpu_worker_get_type(workerId) == STARPU_CPU_WORKER)  || (starpu_worker_get_type(workerId) == STARPU_CUDA_WORKER))
	{
		starpu_pthread_mutex_lock(&data->hp_mutex);
		starpu_pthread_mutex_lock(&data->ready_queues_mutex);
		//pop the first task from its own workers queue
		workerId = starpu_worker_get_id();
		struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerId];
		struct starpu_task *task = NULL;

		task = starpu_st_fifo_taskq_pop_local_task(fifo);
		if(task != NULL)
		{
			unsigned index = data->presentIndexOfTaskInExecution[workerId];
			data->presentTaskInExecution[workerId][index] = task;
			data->presentIndexOfTaskInExecution[workerId] = (index+1)%STARPU_MAX_PIPELINE;
			//double model = task->predicted;
			//fifo->exp_start  = STARPU_MAX(fifo->exp_start, starpu_timing_now()) + model;
			//fifo->exp_len  -= model;
		}
		starpu_pthread_mutex_unlock(&data->ready_queues_mutex);
		starpu_pthread_mutex_unlock(&data->hp_mutex);
		return task;
	}
	else
	{
		/* Workers other than CPUs and GPUs*/
		return NULL;
	}
}

/* hp_lookahead_pre_exec_hook is called right after the data transfer is done and right
 * before the computation to begin, it is useful to update more precisely the
 * value of the expected start, end, length, etc... */
static void hp_lookahead_pre_exec_hook(struct starpu_task *task, unsigned int sched_ctx_id)
{
//    unsigned sched_ctx_id = starpu_sched_ctx_get_ctx_for_task(task);
	int workerId = starpu_worker_get_id();
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerId];
	double model = task->predicted;

	starpu_pthread_mutex_lock(&data->hp_mutex);

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerId, &sched_mutex, &sched_cond);

	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
	if(!isnan(model))
	{
		/* We now modify the prediction */
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_len  -= model;
		data->isTaskInExecution[workerId] = 1;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);

	starpu_pthread_mutex_unlock(&data->hp_mutex);
}

static void hp_lookahead_post_exec_hook(struct starpu_task *task, unsigned int sched_ctx_id)
{
//    unsigned sched_ctx_id = starpu_sched_ctx_get_ctx_for_task(task);
	int workerId = starpu_worker_get_id();
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_pthread_mutex_lock(&data->hp_mutex);
	unsigned pipelineIndex;
	for(pipelineIndex=0; pipelineIndex<STARPU_MAX_PIPELINE; pipelineIndex++)
	{
		if(data->presentTaskInExecution[workerId][pipelineIndex] == task)
		{
			data->presentTaskInExecution[workerId][pipelineIndex] = NULL;
			break;
		}
	}
	STARPU_ASSERT(pipelineIndex != STARPU_MAX_PIPELINE);

	/* release all dependent jobs */
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	/* Remove ourself from the graph before notifying dependencies */
	if (_starpu_graph_record)
		_starpu_graph_drop_job(j);

	_starpu_notify_dependencies(j);

	struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerId];
	double model = task->predicted;

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerId, &sched_mutex, &sched_cond);

	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
	if(!isnan(model))
	{
		/* We now modify the prediction */
		fifo->exp_start = starpu_timing_now();
		data->isTaskInExecution[workerId] = 0;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
	starpu_pthread_mutex_unlock(&data->hp_mutex);
}

struct starpu_sched_policy _starpu_sched_hp_lookahead_policy =
{
	.init_sched = init_hp_lookahead_sched,
	.deinit_sched = deinit_hp_lookahead_sched,
	.add_workers = hp_lookahead_add_workers ,
	.remove_workers = hp_lookahead_remove_workers,
	.push_task = push_task_in_to_hp_lookahead_ready_queue,
	.pop_task = pop_task_from_hp_lookahead_ready_queue,
	.pre_exec_hook = hp_lookahead_pre_exec_hook,
	.post_exec_hook = hp_lookahead_post_exec_hook,
	.policy_name = "hp-lookahead",
	.policy_description = "heteroprio strategy with lookahead simulation",
	.worker_type = STARPU_WORKER_LIST,
};
