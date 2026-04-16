/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <schedulers/starpu_hplookahead_sync_graphs.h>
#include <core/task.h>
#include <common/graph.h>
#include <common/utils.h>
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

//Tunable parameter -- depends on structure of the task graph and platform
#define TCPU 2
#define TGPU 8

/*TODO: May need one extra dimension -- not to rely on constant number of entries */
#define MAXACTIVEWORKERS 30
#define MAXTASKSINWORKERQUEUE 200
#define STARPU_MAX_PIPELINE 4

extern int _starpu_graph_record;
/* Each type of task will have a different queue, sorted in decreaing order of CPU affinity */
struct _starpu_hplookahead_sync_graphs_task_queue
{
	/* codelet would be unique id for each type of tasks*/
	struct starpu_codelet *task_codelet;
	/* queue for each type of tasks*/
	struct starpu_st_fifo_taskq *tasks_queue;
	/* accelration factor for tasks of this queue (CPU time / GPU time)*/
	double acceleration_factor;
};

/* To store the tasks which become ready during simulation*/
struct _starpu_hplookahead_sync_graphs_virtual_node_queues_table
{
	struct starpu_codelet *task_codelet;
	struct _starpu_graph_node*  nodes_queue[STARPU_HPLOOKAHEAD_SYNC_GRAPHS_NTASKSPERQUEUEINSIMULATION];
	/* both indices should be set to 0 before simulation starts */
	int start_index;
	int close_index;
};

/* TODO: Store ready queues also as a list of nodes */
/* To store the tasks which become ready during simulation*/
struct _starpu_hplookahead_sync_graphs_virtual_task_queues_table
{
	struct starpu_codelet *task_codelet;
	struct starpu_task *tasks_queue[STARPU_HPLOOKAHEAD_SYNC_GRAPHS_NTASKSPERQUEUEINSIMULATION];
	/* both indices should be set to 0 before simulation starts */
	int start_index;
	int close_index;
};

struct _starpu_hplookahead_sync_graphs_ready_queue
{
	unsigned  types_of_tasks;
	unsigned  ntasks;
	struct _starpu_hplookahead_sync_graphs_task_queue task_queues[STARPU_HPLOOKAHEAD_SYNC_GRAPHS_MAXTYPESOFTASKS];
//    starpu_pthread_mutex_t hp_mutex;
};

struct _starpu_hp_lookahead_data
{
	/* Data structure for ready queues*/
	struct _starpu_hplookahead_sync_graphs_ready_queue *ready_queues;
	/* Allocated ready tasks to each worker*/
	struct starpu_st_fifo_taskq **workers_queue;

	/* Intermediate table to store ready tasks*/
	struct _starpu_hplookahead_sync_graphs_virtual_task_queues_table intermediate_ready_queues[STARPU_HPLOOKAHEAD_SYNC_GRAPHS_MAXTYPESOFTASKS];
	/* Virtual table to analyze non ready nodes in node graph*/
	struct _starpu_hplookahead_sync_graphs_virtual_node_queues_table virtual_ready_queues[STARPU_HPLOOKAHEAD_SYNC_GRAPHS_MAXTYPESOFTASKS];

	/* Number of different types of tasks seen by scheduler*/
	unsigned types_of_tasks_in_simulation;

	/*TODO: use a separate thread to perform scheduling*/
	/* One dedicated CPU worker to perform scheduling */
	unsigned schedulerWorker;

	/*An array variable holds the expected start time on each worker */
	double *exp_start_in_simulation;
	/*A list of tasks to perform advanced simulation*/
	struct _starpu_graph_node **presentNodeInSimulation;
	/*A list of nodes corresponding to unfinished tasks (to fulfill GPU pipeline) issued to different workers*/
	struct _starpu_graph_node ***presentNodeInExecution;
	/* Expected computation time of tasks already dispatched (in pipeline buffer) to different workers*/
	double **expectedCompletionTimeOfDispatchedTasks;
	/* Index pointing to a valid Node in execution*/
	unsigned *presentIndexOfNodeInExecution;

	/* A very ipmortant field to restrict expected start time of each queue during advanced simulation*/
	/* set and reset by pre and post exec functions*/
	unsigned *isTaskInExecution;

	/* Number of GPUs: needed to determine fair allocations*/
	unsigned nGPUs;

	/* Variable to store number of active workers */
	unsigned nWorkers;
	/* Local indices to worker id mapping */
	unsigned int *localIndicesToWorkerId;

	/* A table to store tasks of worker queues*/
	/* TODO: allocate at runtime in add worker function */
	struct _starpu_graph_node *workersQueue[MAXACTIVEWORKERS][MAXTASKSINWORKERQUEUE];
	unsigned *nEntriesInWQs, *nExaminedEntriesInWQs;

	/* An array of double to store the expected completion
	 * time of scheduled tasks on different resources*/

	/* This temporary buffer is necessary to perform simulation
	 * because some nodes of dispatched tasks (tasks in pipeline buffer)
	 * may point to NULL
	 * */
	double scheduledTaskTimings[MAXACTIVEWORKERS][MAXTASKSINWORKERQUEUE];

	/*mutex variable to protect ready queues */
	starpu_pthread_mutex_t ready_queues_mutex;

	/* mutex variable to protect entries of tasks presently in execution */
	starpu_pthread_mutex_t protectPipelineEntries;
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
/* Display content of virtual table  -- useful for debug purpose */
static void displayVT(struct _starpu_hp_lookahead_data *data)
{
	unsigned i;
	for(i=0; i<data->types_of_tasks_in_simulation; i++)
	{
		int j;
		_STARPU_DEBUG("VQ %d(startIndex=%d,endIndex=%d):", i, data->virtual_ready_queues[i].start_index, data->virtual_ready_queues[i].close_index);
		for(j=data->virtual_ready_queues[i].start_index; j<data->virtual_ready_queues[i].close_index; j++)
		{
			_STARPU_DEBUG(" %llx", data->virtual_ready_queues[i].nodes_queue[j]->job->task->tag_id);
		}
		_STARPU_DEBUG("\n");
	}
}
#endif

/* Add the task corresponding to the given node in the virtual table */
static inline void addToVirtualqueue(struct _starpu_graph_node *node, struct _starpu_hp_lookahead_data *data)
{
	struct starpu_task *task = node->job->task;
	unsigned i;
	for(i=0; i<data->types_of_tasks_in_simulation; i++)
	{
		if(data->virtual_ready_queues[i].task_codelet == task->cl)
		{
			int endIndex = data->virtual_ready_queues[i].close_index;
			data->virtual_ready_queues[i].nodes_queue[endIndex] = node;
			data->virtual_ready_queues[i].close_index++;
			return;
		}
	}

	/* A new ttype of task has been found in simulation -- make a separate entry for this task*/
	data->virtual_ready_queues[i].task_codelet = task->cl;
	data->virtual_ready_queues[i].nodes_queue[0] = node;
	data->virtual_ready_queues[i].start_index = 0;
	data->virtual_ready_queues[i].close_index = 1;
	data->types_of_tasks_in_simulation++;
	return;
}

/* Add task corresponding to the given node in the virtual table
 *  and  release its successors whose all dependencies are met in simulation */
static void releaseAndAddtoVT(struct _starpu_graph_node *node, struct _starpu_hp_lookahead_data *data, struct starpu_task *taskInConsideration)
{
	/*
	  if(node->job)
	  _STARPU_DEBUG("Before adding successors of task %llx\n", node->job->task->tag_id);
	  displayVT(data);
	*/

	unsigned int i;
	for(i=0; i<node->n_outgoing; i++)
	{
		struct _starpu_graph_node *nodeChild = node->outgoing[i];

		if(!nodeChild)
			continue;

		unsigned l;
		unsigned alreadyVisited = 0;
		for(l=0; l<i; l++)
		{
			if(node->outgoing[l] == nodeChild)
			{
				alreadyVisited = 1;
				break;
			}
		}
		if(alreadyVisited == 1)
			continue;

		/* Check whether this task already has seen in ready queue*/

		if(nodeChild->presentTaskInSimulation != taskInConsideration)
		{
			/* Number of remaining incoming edges */
			unsigned k;
			unsigned remainingEdges = 0;
			for(k=0; k<nodeChild->n_incoming; k++)
			{
				if(!nodeChild->incoming[k])
					continue;
				remainingEdges ++;
			}
			nodeChild->presentTaskInSimulation = taskInConsideration;
			nodeChild->ndeps_remaining_in_simulation = remainingEdges;
		}

		/*One task may create more than one dependency to another task*/
		unsigned k;
		for(k=0; k<nodeChild->n_incoming; k++)
			if(nodeChild->incoming[k] == node)
				nodeChild->ndeps_remaining_in_simulation--;

		if(nodeChild->ndeps_remaining_in_simulation == 0)
		{
			if((nodeChild->job != NULL) && (nodeChild->job->task->cl != NULL))
				addToVirtualqueue(nodeChild, data);
			else
				releaseAndAddtoVT(nodeChild, data, taskInConsideration);
		}
	}

	/*
	  if(node->job)
	  _STARPU_DEBUG("After adding successors of task %llx\n", node->job->task->tag_id);
	  displayVT(data);
	*/
}

static struct starpu_task* pop_task_from_hp_lookahead_ready_queue(unsigned sched_ctx_id)
{
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned workerId = starpu_worker_get_id();

	/* TODO: Instead of dedicate a worker, create a dedicate thread to perform scheduling */

	if(workerId == data->schedulerWorker)
	{
		//Algorithm:
		//Step A:
		// 1) if ready task is non empty, scheduling worker will try to schedule some tasks
		// 2) lock node_graph
		// 3) lock droplist
		// 3) synchronize task graph
		// 4) Copy start time and all the scheduled non completed nodes
		// 5) unlock droplist

		/* ready_queues is a shared data structure, but computational threads can only increase the ntasks field.  Thereofre, It is safe to test this variable without using any loop */
		if(data->ready_queues->ntasks == 0)
		{
			return NULL;
		}

		/* A big lock : may impact the performance, but required for advance lookup*/
		/* lock node graph*/
		_starpu_graph_wrlock();

		_starpu_drop_lock();
		synchronize_node_and_task_graphs();

		/* Store expected start time of each worker and present task in execution into temporary buffer */
		unsigned worker;
		double minCompleteionTimeOnWorkers = DBL_MAX;
		int workerIdOfMinLoadedWorker = -1;

		for(worker =0; worker < data->nWorkers; worker++)
		{
			unsigned originalWorker = data->localIndicesToWorkerId[worker];
			if(originalWorker == data->schedulerWorker)
				continue;
			struct starpu_st_fifo_taskq *fifo = data->workers_queue[originalWorker];
			starpu_pthread_mutex_t *sched_mutex;
			starpu_pthread_cond_t *sched_cond;
			starpu_worker_get_sched_condition(originalWorker, &sched_mutex, &sched_cond);
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);

			starpu_pthread_mutex_lock(&data->protectPipelineEntries);

			if(data->isTaskInExecution[originalWorker] == 0)
				fifo->exp_start = starpu_timing_now();

			data->exp_start_in_simulation[worker] = fifo->exp_start;

			double completionTimeOnThisWorker = fifo->exp_start + fifo->exp_len;
			if(completionTimeOnThisWorker < minCompleteionTimeOnWorkers)
			{
				minCompleteionTimeOnWorkers = completionTimeOnThisWorker;
				workerIdOfMinLoadedWorker = originalWorker;
			}
			else if((completionTimeOnThisWorker == minCompleteionTimeOnWorkers) &&(starpu_worker_get_type(originalWorker) == STARPU_CUDA_WORKER))
			{
				workerIdOfMinLoadedWorker = originalWorker;
			}

			//TODO: move it when workerIdOfMinLoaded worker corresponds to CPU worker
			/* Copy tasks of pipeline buffer and their timings to dedicated data structure */
			unsigned counter = 0;
			unsigned index = data->presentIndexOfNodeInExecution[originalWorker];
			struct _starpu_graph_node *taskNode = data->presentNodeInExecution[originalWorker][index];
			if(taskNode != NULL)
			{
				data->workersQueue[worker][counter] = taskNode;
				data->scheduledTaskTimings[worker][counter] = data->expectedCompletionTimeOfDispatchedTasks[originalWorker][index];
				counter++;
			}
			unsigned j;
			for(j = (index+1)% STARPU_MAX_PIPELINE; j!= index; j = (j+1)%STARPU_MAX_PIPELINE)
			{
				taskNode = data->presentNodeInExecution[originalWorker][j];
				if(taskNode != NULL)
				{
					data->workersQueue[worker][counter] = taskNode;
					data->scheduledTaskTimings[worker][counter] = data->expectedCompletionTimeOfDispatchedTasks[originalWorker][j];
					counter++;
				}
			}

			struct starpu_task *task = starpu_task_list_front(&fifo->taskq);

			while(task)
			{
				struct _starpu_job *j2 = _starpu_get_job_associated_to_task(task);
				data->workersQueue[worker][counter] = j2->graph_node;
				data->scheduledTaskTimings[worker][counter] = task->predicted;
				task = task->next;
				counter++;
			}
			data->nEntriesInWQs[worker] = counter;
			data->nExaminedEntriesInWQs[worker] = 0;
			starpu_pthread_mutex_unlock(&data->protectPipelineEntries);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
        }

		/* Release lock on drop list: synchronization has been completed and all dispatched tasks on workers have been copied to a temporary buffer*/
		_starpu_drop_unlock();

		/* There should be some valid worker Id
		 * who is going to be idle soon*/
		STARPU_ASSERT(workerIdOfMinLoadedWorker != -1);
		/* Copy all scheduled nodes into a temporary structure */

		struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerIdOfMinLoadedWorker];
		if((starpu_worker_get_type(workerIdOfMinLoadedWorker) == STARPU_CUDA_WORKER) && (fifo->ntasks <= TGPU))
		{
			/* A GPU worker pops fair number of tasks from ready queues and
			 * puts in its worker queue */
			int nGPUs = data->nGPUs;
			int nFairAllocations = (data->ready_queues->ntasks + nGPUs-1)/nGPUs;
			while((nFairAllocations > 0) && (fifo->ntasks <= TGPU))
			{
				// while conditions ensure that GPU gets a nonnull task
				int queueId;
				struct starpu_task *task = NULL;
				starpu_pthread_mutex_lock(&data->ready_queues_mutex);
				for(queueId=data->ready_queues->types_of_tasks - 1; queueId >=0 && task == NULL; queueId--)
					task = starpu_st_fifo_taskq_pop_local_task(data->ready_queues->task_queues[queueId].tasks_queue);

				data->ready_queues->ntasks --;
				starpu_pthread_mutex_unlock(&data->ready_queues_mutex);

				struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerIdOfMinLoadedWorker, sched_ctx_id);

				/* Expected duration of task on current worker */
				double model = starpu_task_expected_length(task, perf_arch, 0);
				task->predicted = model;

				unsigned memory_node = starpu_worker_get_memory_node(workerIdOfMinLoadedWorker);
				starpu_prefetch_task_input_on_node(task, memory_node);

				starpu_pthread_mutex_t *sched_mutex;
				starpu_pthread_cond_t *sched_cond;
				starpu_worker_get_sched_condition(workerIdOfMinLoadedWorker, &sched_mutex, &sched_cond);
				STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
				fifo->exp_len  += model;
				starpu_st_fifo_taskq_push_task(fifo, task);
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);

				nFairAllocations--;
			}
		}
		else if((starpu_worker_get_type(workerIdOfMinLoadedWorker) == STARPU_CPU_WORKER) && (fifo->ntasks <= TCPU))
		{
			unsigned queueId;
			struct starpu_task *taskToBeScheduled = NULL;

			starpu_pthread_mutex_lock(&data->ready_queues_mutex);
			for(queueId=0; queueId<data->ready_queues->types_of_tasks && taskToBeScheduled == NULL; queueId++)
				taskToBeScheduled = starpu_st_fifo_taskq_pop_local_task(data->ready_queues->task_queues[queueId].tasks_queue);
			/*Control comes here which ensures that taskToBeScheduled would not be NULL*/
			STARPU_ASSERT(taskToBeScheduled != NULL);
			data->ready_queues->ntasks --;

			//_STARPU_DEBUG("Try to schedule Task %lx at line number %d\n",  taskToBeScheduled->tag_id, __LINE__);

			/* Copy ready queues to a temporary buffer*/
			for(queueId=0; queueId<data->ready_queues->types_of_tasks; queueId++)
			{
				data->intermediate_ready_queues[queueId].task_codelet = data->ready_queues->task_queues[queueId].task_codelet;
				data->intermediate_ready_queues[queueId].start_index = 0;
				unsigned counter = 0;
				struct starpu_task *task = starpu_task_list_front(&data->ready_queues->task_queues[queueId].tasks_queue->taskq);
				while(task != NULL)
				{
					data->intermediate_ready_queues[queueId].tasks_queue[counter] = task;
					counter++;
					task = task ->next;
				}
				data->intermediate_ready_queues[queueId].close_index = counter;
			}

			starpu_pthread_mutex_unlock(&data->ready_queues_mutex);

			/* Add taskToBeScheduled to the worker that has minimum load*/
			{
				unsigned virtualWorkerIdOfMinLoadedWorker = getVirtualIndex(workerIdOfMinLoadedWorker, data->localIndicesToWorkerId, data->nWorkers);
				struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerIdOfMinLoadedWorker, sched_ctx_id);

				/* Expected duration of task on current worker */
				double model = starpu_task_expected_length(taskToBeScheduled, perf_arch, 0);
				taskToBeScheduled->predicted = model;

				unsigned counter = data->nEntriesInWQs[virtualWorkerIdOfMinLoadedWorker];
				struct _starpu_job *j = _starpu_get_job_associated_to_task(taskToBeScheduled);
				data->workersQueue[virtualWorkerIdOfMinLoadedWorker][counter] = j->graph_node;
				data->scheduledTaskTimings[virtualWorkerIdOfMinLoadedWorker][counter] = model;
				data->nEntriesInWQs[virtualWorkerIdOfMinLoadedWorker] = counter + 1;
			}

			/*Copy the codelet information from ready queues to virtual table */
			/*Optimization: It can be assigned when a new entry is created in ready queues */
			{
				data->types_of_tasks_in_simulation = data->ready_queues->types_of_tasks;
				for(queueId=0; queueId<data->types_of_tasks_in_simulation; queueId++)
				{
					data->virtual_ready_queues[queueId].task_codelet = data->ready_queues->task_queues[queueId].task_codelet;
					data->virtual_ready_queues[queueId].start_index = 0;
					data->virtual_ready_queues[queueId].close_index = 0;
				}
			}
			/*
			 * Release all tasks whose expect completion time is
			 * not more than minCompleteionTimeOnWorkers
			 *
			 */
			for(worker =0; worker < data->nWorkers; worker++)
			{
				if(data->localIndicesToWorkerId[worker] == data->schedulerWorker)
					continue;
				unsigned index = data->nExaminedEntriesInWQs[worker];
				while(index < data->nEntriesInWQs[worker])
				{
					double expExecutionTime = data->scheduledTaskTimings[worker][index];
					if(data->exp_start_in_simulation[worker] + expExecutionTime > minCompleteionTimeOnWorkers)
						break;
					data->exp_start_in_simulation[worker] = data->exp_start_in_simulation[worker] + expExecutionTime;
					releaseAndAddtoVT(data->workersQueue[worker][index], data, taskToBeScheduled);
					index++;
				}
				if(index < data->nEntriesInWQs[worker])
				{
					data->presentNodeInSimulation[worker] = data->workersQueue[worker][index];
					data->exp_start_in_simulation[worker] = data->exp_start_in_simulation[worker] + data->scheduledTaskTimings[worker][index];
					index++;
				}
				else
				{
					data->presentNodeInSimulation[worker] = NULL;
				}
				data->nExaminedEntriesInWQs[worker] = index;
			}

			/* All temporary structures are set */
			/* Start the simulation until final decision for task taskInConsideration is taken */
		repeatWhileTrue:
			while(1)
			{
				/* Find the worker who is going to be idle soon */
				/* In case of tie between CPU and GPU, GPU would be preferred */
				double minStartTime =  DBL_MAX;
				unsigned minIndex;
				for(worker=0; worker < data->nWorkers; worker++)
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

				/* Release all tasks finished at minStartTime */
				for(worker=0; worker < data->nWorkers; worker++)
				{
					if((data->exp_start_in_simulation[worker] == minStartTime) && (data->presentNodeInSimulation[worker] != NULL))
					{
						if((data->presentNodeInSimulation[worker]->job != NULL)&&(data->presentNodeInSimulation[worker]->job->task == taskToBeScheduled))
						{
							unsigned originalWorker = data->localIndicesToWorkerId[worker];
							struct starpu_st_fifo_taskq *fifo2 = data->workers_queue[originalWorker];
							struct starpu_perfmodel_arch *perf_arch = starpu_worker_get_perf_archtype(originalWorker, sched_ctx_id);

							/* Expected duration of task on current worker */
							double model = starpu_task_expected_length(taskToBeScheduled, perf_arch, 0);
							taskToBeScheduled->predicted = model;

							unsigned memory_node = starpu_worker_get_memory_node(originalWorker);
							starpu_prefetch_task_input_on_node(taskToBeScheduled, memory_node);

							starpu_pthread_mutex_t *sched_mutex;
							starpu_pthread_cond_t *sched_cond;
							starpu_worker_get_sched_condition(originalWorker, &sched_mutex, &sched_cond);
							STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
							fifo2->exp_len  += model;
							starpu_st_fifo_taskq_push_task(fifo2, taskToBeScheduled);
							STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);

							// Terminate the simulation
							goto afterAnalyzing;
						}
						else
						{
							/* Release tasks which become ready after completion of this tasks and add to virtual simulation table */
							releaseAndAddtoVT(data->presentNodeInSimulation[worker], data, taskToBeScheduled);
							data->presentNodeInSimulation[worker] = NULL;
						}
					}
				}

				/* Select a task from workers queue */
				if(data->nExaminedEntriesInWQs[minIndex] < data->nEntriesInWQs[minIndex])
				{
					int index = data->nExaminedEntriesInWQs[minIndex];
					data->presentNodeInSimulation[minIndex] = data->workersQueue[minIndex][index];
					data->exp_start_in_simulation[minIndex] = data->exp_start_in_simulation[minIndex] + data->scheduledTaskTimings[minIndex][index];
					data->nExaminedEntriesInWQs[minIndex] = index + 1;
					goto repeatWhileTrue;
				}

				/* Set the execution of task  present on the worker with minIndex Id */
				data->presentNodeInSimulation[minIndex] = NULL;
				unsigned originalMinIndex = data->localIndicesToWorkerId[minIndex];
				/* No task is in the selected worker's queue */
				/* Select a task from virtual_ready_queues(nodes) or intermediate_ready_queues (tasks) */
				if(starpu_worker_get_type(originalMinIndex) == STARPU_CPU_WORKER)
				{
					unsigned i;
					int found=0;
					for(i=0; (i<data->types_of_tasks_in_simulation) && (found == 0); i++)
					{
						//TODO: Better to consider the last task of the queue (assuming priortized tasks are stored at the beginning of the queue)
						if(data->virtual_ready_queues[i].start_index < data->virtual_ready_queues[i].close_index)
						{
							/* There are some tasks in the virtual queue for ith task type */
							int startIndexInVirtualQueue = data->virtual_ready_queues[i].start_index;
							struct _starpu_graph_node *nodeInVirtualQueue = data->virtual_ready_queues[i].nodes_queue[startIndexInVirtualQueue];
							struct _starpu_graph_node *nodeInReadyQueue = NULL;
							while(data->intermediate_ready_queues[i].start_index < data->intermediate_ready_queues[i].close_index)
							{
								int startIndexInReadyQueue = data->intermediate_ready_queues[i].start_index;
								struct starpu_task *task = data->intermediate_ready_queues[i].tasks_queue[startIndexInReadyQueue];
								struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

								if(j->graph_node->presentTaskInSimulation != taskToBeScheduled)
								{
									nodeInReadyQueue = j->graph_node;
									break;
								}
								data->intermediate_ready_queues[i].start_index ++;
							}
							struct _starpu_graph_node *node = NULL;
							if((nodeInReadyQueue == NULL) || (nodeInReadyQueue->job->task->priority < nodeInVirtualQueue->job->task->priority))
							{
								node = nodeInVirtualQueue;
								data->virtual_ready_queues[i].start_index++;
							}
							else
							{
								node = nodeInReadyQueue;
								data->intermediate_ready_queues[i].start_index ++;
							}
							struct starpu_task *task = node->job->task;
							struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
							/* Expected duration of task on current worker */
							double model = starpu_task_expected_length(task, perf_arch, 0);
							data->exp_start_in_simulation[minIndex] += model;
							data->presentNodeInSimulation[minIndex] = node;

							found = 1;
						}
						else
						{
							/* find a task from a ready queue*/
							while(data->intermediate_ready_queues[i].start_index < data->intermediate_ready_queues[i].close_index)
							{
								int startIndexInReadyQueue = data->intermediate_ready_queues[i].start_index;
								struct starpu_task *task = data->intermediate_ready_queues[i].tasks_queue[startIndexInReadyQueue];
								struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
								data->intermediate_ready_queues[i].start_index ++;
								if(j->graph_node->presentTaskInSimulation != taskToBeScheduled)
								{
									struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
									/* Expected duration of task on current worker */
									double model = starpu_task_expected_length(task, perf_arch, 0);
									data->exp_start_in_simulation[minIndex] += model;
									data->presentNodeInSimulation[minIndex] = j->graph_node;

									found = 1;
									break;
								}
							}
						}
					}
					if(found == 1)
						goto repeatWhileTrue;

					/* If control comes here: It indicates that there is not any ready task to schedule at this moment */
					/* Find the next eventpoint and update the exp_start time for each worker appropriately */
					{
						double nextEventPoint = DBL_MAX;
						/* Proceed to next iteration find a new eventpoint */
						for(worker = 0; worker <data->nWorkers; worker++)
						{
							unsigned originalWorker = data->localIndicesToWorkerId[worker];

							if(originalWorker == data->schedulerWorker)
								continue;

							if((data->presentNodeInSimulation[worker] != NULL) && (data->exp_start_in_simulation[worker] < nextEventPoint))
							{
								nextEventPoint = data->exp_start_in_simulation[worker];
							}
						}

						/* Set next eventpoint for all idle workers */
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
						int i;
						int found=0;
						for(i = data->types_of_tasks_in_simulation-1; (i>=0) && (found == 0); i--)
						{
							if(data->virtual_ready_queues[i].start_index < data->virtual_ready_queues[i].close_index)
							{
								/* There are some tasks in the virtual queue  for ith task type */
								int startIndexInVirtualQueue = data->virtual_ready_queues[i].start_index;
								struct _starpu_graph_node *nodeInVirtualQueue = data->virtual_ready_queues[i].nodes_queue[startIndexInVirtualQueue];
								struct _starpu_graph_node *nodeInReadyQueue = NULL;
								while(data->intermediate_ready_queues[i].start_index < data->intermediate_ready_queues[i].close_index)
								{
									int startIndexInReadyQueue = data->intermediate_ready_queues[i].start_index;
									struct starpu_task *task = data->intermediate_ready_queues[i].tasks_queue[startIndexInReadyQueue];
									struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

									if(j->graph_node->presentTaskInSimulation != taskToBeScheduled)
									{
										nodeInReadyQueue = j->graph_node;
										break;
									}
									data->intermediate_ready_queues[i].start_index ++;
								}
								struct _starpu_graph_node *node = NULL;
								if((nodeInReadyQueue == NULL) || (nodeInReadyQueue->job->task->priority < nodeInVirtualQueue->job->task->priority))
								{
									node = nodeInVirtualQueue;
									data->virtual_ready_queues[i].start_index++;
								}
								else
								{
									node = nodeInReadyQueue;
									data->intermediate_ready_queues[i].start_index ++;
								}
								struct starpu_task *task = node->job->task;
								struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
								/* Expected duration of task on current worker */
								double model = starpu_task_expected_length(task, perf_arch, 0);
								data->exp_start_in_simulation[minIndex] += model;
								data->presentNodeInSimulation[minIndex] = node;

								found = 1;
							}
							else
							{
								/* Find a task from a ready queue*/
								while(data->intermediate_ready_queues[i].start_index < data->intermediate_ready_queues[i].close_index)
								{
									int startIndexInReadyQueue = data->intermediate_ready_queues[i].start_index;
									struct starpu_task *task = data->intermediate_ready_queues[i].tasks_queue[startIndexInReadyQueue];
									struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
									data->intermediate_ready_queues[i].start_index ++;
									if(j->graph_node->presentTaskInSimulation != taskToBeScheduled)
									{
										struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
										/* Expected duration of task on current worker */
										double model = starpu_task_expected_length(task, perf_arch, 0);
										data->exp_start_in_simulation[minIndex] += model;
										data->presentNodeInSimulation[minIndex] = j->graph_node;

										found = 1;
										break;
									}
								}
							}
						}
						if(found == 1)
							goto repeatWhileTrue;
					}
					{
						/* If control comes here, it indicates that some GPU is idle*/
						/* calculate expected end time of taskInConsideration task on minIndex and
						 * compare with its expected completion time on CPU worker  */
						int cpuWorkerId = -1;
						for(worker=0; worker < data->nWorkers; worker++)
						{
							/* scheduler worker will have NULL in current Task(as well as in simulation) in execution */
							if((data->presentNodeInSimulation[worker] != NULL) && ((data->presentNodeInSimulation[worker]->job != NULL) && data->presentNodeInSimulation[worker]->job->task == taskToBeScheduled))
							{
								cpuWorkerId = worker;
								break;
							}
						}
						STARPU_ASSERT_MSG(cpuWorkerId != -1, "Task in consideration is not running on CPU: something strange");

						unsigned originalMinIndex = data->localIndicesToWorkerId[minIndex];
						struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalMinIndex, sched_ctx_id);
						/* Expected duration of task on current worker */
						double model = starpu_task_expected_length(taskToBeScheduled, perf_arch, 0);
						unsigned originalWorker = -1;
						if(data->exp_start_in_simulation[minIndex] + model < data->exp_start_in_simulation[cpuWorkerId])
						{
							originalWorker = data->localIndicesToWorkerId[minIndex];
						}
						else
						{
							/* Place task in the cpu worker queue */
							/* One optimization could be schedule all ready tasks (even non ready tasks) involved in for decision of taskInConsideration*/

							originalWorker = data->localIndicesToWorkerId[cpuWorkerId];
						}
						//Take decision for  task in consideration: taskToBeScheduled
						{
							struct starpu_st_fifo_taskq *fifo = data->workers_queue[originalWorker];

							struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(originalWorker, sched_ctx_id);

							/* Expected duration of task on current worker */
							double model = starpu_task_expected_length(taskToBeScheduled, perf_arch, 0);
							taskToBeScheduled->predicted = model;

							unsigned memory_node = starpu_worker_get_memory_node(originalWorker);
							starpu_prefetch_task_input_on_node(taskToBeScheduled, memory_node);

							starpu_pthread_mutex_t *sched_mutex;
							starpu_pthread_cond_t *sched_cond;
							starpu_worker_get_sched_condition(originalWorker, &sched_mutex, &sched_cond);
							STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
							fifo->exp_len  += model;
							starpu_st_fifo_taskq_push_task(fifo, taskToBeScheduled);
							STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
						}

						goto afterAnalyzing;
					}
				}
				/* end of While true loop */
			}
		}

	afterAnalyzing:
		/* Scheduling decison has been made
		 * unlock node graph */
		_starpu_graph_wrunlock();

		/* Scheduler worker won't execute any task*/
		return NULL;
	}
	else if((starpu_worker_get_type(workerId) == STARPU_CPU_WORKER)  || (starpu_worker_get_type(workerId) == STARPU_CUDA_WORKER))
	{
		workerId = starpu_worker_get_id();
		struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerId];
		struct starpu_task *task = NULL;

		task = starpu_st_fifo_taskq_pop_local_task(fifo);
		if(task != NULL)
		{
			unsigned index = data->presentIndexOfNodeInExecution[workerId];
			struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

			starpu_pthread_mutex_lock(&data->protectPipelineEntries);

			data->presentNodeInExecution[workerId][index] = j->graph_node;
			data->expectedCompletionTimeOfDispatchedTasks[workerId][index] = task->predicted;
			data->presentIndexOfNodeInExecution[workerId] = (index+1)%STARPU_MAX_PIPELINE;
			starpu_pthread_mutex_unlock(&data->protectPipelineEntries);
		}
		return task;
	}
	else
	{
		/* Workers other than CPUs and GPUs*/
		return NULL;
	}
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
					/* No one on that queue may execute this task */
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
					/* No one on that queue may execute this task */
					continue;
				}
				double expected_length = starpu_task_expected_length(task, perf_arch, nimpl);
				if(expected_length < gpu_time)
					gpu_time = expected_length;
			}
		}
		else
			/* Other than CPU/GPU worker */
			continue;
	}
	/* To avoid division by zero in some undesirable situation
	 * add 1 to denominator
	 */
	return cpu_time/(gpu_time + 1.0);
}

static int  push_task_in_to_hp_lookahead_ready_queue(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
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
		unsigned chosen_impl;
		unsigned worker;
		unsigned min_ntasks;
		unsigned impl_mask;

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
		return 0;
    }

	starpu_pthread_mutex_lock(&data->ready_queues_mutex);
	/* Insert task in the appropriate queue */
	unsigned int queueId;
	for(queueId=0; queueId<data->ready_queues->types_of_tasks; queueId++)
	{
		/* Checking whether some task corresponding to the same codelet has been
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
		// First instance of this type of task
		/* Create a new ready queue to store this type of tasks */
		struct starpu_st_fifo_taskq *current_taskq = starpu_st_fifo_taskq_create();
		starpu_st_fifo_taskq_push_task(current_taskq, task);
		// Compute acceleration factor for this type of tasks
		double current_acceleration_factor = compute_acceleration_ratio(task, sched_ctx_id);
		/* Insert the queues based on the nondecreasing order of acceleration factor */
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

	starpu_pthread_mutex_unlock(&data->ready_queues_mutex);

	return 0;
}

/* hp_lookahead_pre_exec_hook is called right after the data transfer is done and right
 * before the computation to begin, it is useful to update more precisely the
 * value of the expected start, end, length, etc... */
static void hp_lookahead_pre_exec_hook(struct starpu_task *task, unsigned int sched_ctx_id)
{
	int workerId = starpu_worker_get_id();
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerId];

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerId, &sched_mutex, &sched_cond);

	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
	fifo->exp_start = starpu_timing_now();
	if(task->cl->model != NULL)
		data->isTaskInExecution[workerId] = 1;

	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
}

static void hp_lookahead_post_exec_hook(struct starpu_task *task, unsigned int sched_ctx_id)
{
	//_STARPU_DEBUG("Task=%lx completed at line number %d\n", task->tag_id,  __LINE__);
	int workerId = starpu_worker_get_id();
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned pipelineIndex;
	struct starpu_st_fifo_taskq *fifo = data->workers_queue[workerId];
	double model = task->predicted;

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerId, &sched_mutex, &sched_cond);

	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);

	starpu_pthread_mutex_lock(&data->protectPipelineEntries);

	for(pipelineIndex=0; pipelineIndex<STARPU_MAX_PIPELINE; pipelineIndex++)
	{
		if((data->presentNodeInExecution[workerId][pipelineIndex]) && (data->presentNodeInExecution[workerId][pipelineIndex]->job->task == task))
		{
			data->presentNodeInExecution[workerId][pipelineIndex] = NULL;
			break;
		}
	}
	STARPU_ASSERT(pipelineIndex != STARPU_MAX_PIPELINE);

	if(!isnan(model))
	{
		/* We now modify the prediction */
		fifo->exp_start = starpu_timing_now();
		fifo->exp_len  -= model;
		data->isTaskInExecution[workerId] = 0;
	}

	starpu_pthread_mutex_unlock(&data->protectPipelineEntries);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
}

static void hp_lookahead_add_workers(unsigned sched_ctx_id, int *workerIds, unsigned nworkers)
{
	struct _starpu_hp_lookahead_data *data = (struct _starpu_hp_lookahead_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned nGPUs = 0;

	unsigned workerId;
	unsigned i;

	data->nWorkers = nworkers;
	_STARPU_MALLOC(data->localIndicesToWorkerId, nworkers * sizeof(unsigned));
	_STARPU_MALLOC(data->nEntriesInWQs, nworkers * sizeof(unsigned));
	_STARPU_MALLOC(data->nExaminedEntriesInWQs, nworkers * sizeof(unsigned));

	for (i = 0; i < nworkers; i++)
	{
		workerId = workerIds[i];
		/*TODO: Assumption: order of workers do not change*/
		data->localIndicesToWorkerId[i] = workerId;
		/* If the worker has alreadry belonged to this context
		   the queue and the synchronization variables have been already initialized */
		if(data->workers_queue[workerId] == NULL)
		{
			data->workers_queue[workerId] = starpu_st_fifo_taskq_create();
			data->workers_queue[workerId]->exp_start = 0;
			data->workers_queue[workerId]->exp_len = 0;
		}
		/* Allocate memory to store all tasks in execution to fulfill pipeline length*/
		_STARPU_MALLOC(data->presentNodeInExecution[workerId], STARPU_MAX_PIPELINE * sizeof(struct _starpu_graph_node *));
		_STARPU_MALLOC(data->expectedCompletionTimeOfDispatchedTasks[workerId], STARPU_MAX_PIPELINE * sizeof(double));
		unsigned j;
		for(j=0; j<STARPU_MAX_PIPELINE; j++)
			data->presentNodeInExecution[workerId][j] = NULL;

		data->presentIndexOfNodeInExecution[workerId] = 0;
		if(starpu_worker_get_type(workerId) == STARPU_CUDA_WORKER)
			nGPUs++;
	}
	data->nGPUs = nGPUs;

	for (i = 0; i < nworkers; i++)
	{
		workerId = workerIds[i];
		if(starpu_worker_get_type(workerId) == STARPU_CPU_WORKER)
		{
			/* First CPU worker is doing allocation for all other workers*/
			/* TODO: dedicate one thread for scheduling instead of a compuational worker */
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
		free(data->presentNodeInExecution[workerId]);
		free(data->expectedCompletionTimeOfDispatchedTasks[workerId]);
	}
	free(data->localIndicesToWorkerId);
	free(data->nEntriesInWQs);
	free(data->nExaminedEntriesInWQs);
}

static void init_hp_lookahead_sched(unsigned sched_ctx_id)
{
	struct _starpu_hp_lookahead_data *data;
	_STARPU_MALLOC(data, sizeof(struct _starpu_hp_lookahead_data));
	_STARPU_MALLOC(data->ready_queues, sizeof(struct _starpu_hplookahead_sync_graphs_ready_queue));
	data->ready_queues->types_of_tasks = 0;
	data->ready_queues->ntasks = 0;
	_STARPU_MALLOC(data->workers_queue, STARPU_NMAXWORKERS*sizeof(struct starpu_st_fifo_taskq*));
	_STARPU_MALLOC(data->exp_start_in_simulation, STARPU_NMAXWORKERS*sizeof(double));
	_STARPU_CALLOC(data->presentNodeInSimulation, STARPU_NMAXWORKERS, sizeof(struct _starpu_graph_node  *));
	_STARPU_CALLOC(data->presentNodeInExecution, STARPU_NMAXWORKERS, sizeof(struct _starpu_graph_node **));
	_STARPU_CALLOC(data->expectedCompletionTimeOfDispatchedTasks, STARPU_NMAXWORKERS, sizeof(double*));
	_STARPU_CALLOC(data->presentIndexOfNodeInExecution, STARPU_NMAXWORKERS, sizeof(unsigned));
	/*TODO: Can be changed to a boolean value*/
	_STARPU_MALLOC(data->isTaskInExecution, STARPU_NMAXWORKERS * sizeof(unsigned));
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		data->workers_queue[i] = NULL;
		data->presentNodeInExecution[i] = NULL;
		data->isTaskInExecution[i] = 0;
	}
	/* Create a condition variable to protect it */

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);

	starpu_pthread_mutex_init(&data->ready_queues_mutex, NULL);
	starpu_pthread_mutex_init(&data->protectPipelineEntries, NULL);

	/* Record graph structure of jobs in terms of incoming/outgoing edges */
	_starpu_graph_record = 1;
	_STARPU_DEBUG("Initialising hp-lookahead-sync scheduler\n");
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

	starpu_pthread_mutex_destroy(&data->ready_queues_mutex);
	starpu_pthread_mutex_destroy(&data->protectPipelineEntries);
	free(data->ready_queues);
	free(data->workers_queue);
	free(data->exp_start_in_simulation);
	free(data->presentNodeInExecution);
	free(data->expectedCompletionTimeOfDispatchedTasks);
	free(data->presentIndexOfNodeInExecution);
	free(data->presentNodeInSimulation);
	free(data->isTaskInExecution);
	free(data);

	_STARPU_DEBUG("Deinitializing hp-lookahead-sync scheduler\n");
}

struct starpu_sched_policy _starpu_sched_hp_lookahead_sync_policy =
{
	.init_sched = init_hp_lookahead_sched,
	.deinit_sched = deinit_hp_lookahead_sched,
	.add_workers = hp_lookahead_add_workers ,
	.remove_workers = hp_lookahead_remove_workers,
	.push_task = push_task_in_to_hp_lookahead_ready_queue,
	.pop_task = pop_task_from_hp_lookahead_ready_queue,
	.pre_exec_hook = hp_lookahead_pre_exec_hook,
	.post_exec_hook = hp_lookahead_post_exec_hook,
	.policy_name = "hp-lookahead-sync",
	.policy_description = "heteroprio strategy with lookahead simulation (based on synchronization of node graph)",
	.worker_type = STARPU_WORKER_LIST,
};
