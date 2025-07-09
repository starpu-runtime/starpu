/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2025  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * Record which kinds of tasks have been executed, to later on compute an upper
 * bound of the performance that could have theoretically been achieved
 */

#include <starpu.h>
#include <starpu_config.h>
#include <profiling/bound.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/memory_nodes.h>
#include <sched_policies/splitter.h>
#ifdef STARPU_HAVE_GLPK_H
#include <float.h>
#include <glpk.h>
#endif /* STARPU_HAVE_GLPK_H */

/* TODO: output duration between starpu_bound_start and starpu_bound_stop */

/* TODO: compute critical path and introduce it in the LP */

#ifdef STARPU_RECURSIVE_TASKS
#define NMAX_RECTYPE 16
#define NMAX_LEVEL 4
/*
 * Record without dependencies: just count each kind of task
 *
 * The linear programming problem will just have as variables:
 * - the number of tasks of kind `t' executed by worker `w'
 * - the total duration
 *
 * and the constraints will be:
 * - the time taken by each worker to complete its assigned tasks is lower than
 *   the total duration.
 * - the total number of tasks of a given kind is equal to the number run by the
 *   application.
 */
struct splitter_bound_task_pool
{
	/* Which codelet has been executed */
	struct starpu_codelet *cl;
	/* Task footprint key (for history-based perfmodel) */
	uint32_t footprint;
	/* Number of tasks of this kind */
	unsigned long n;
	/* Other task kinds */
	long handle_sizes; // field used to know if a task is more CPU or GPU related
	volatile struct _starpu_job *ref_pjob; // if a task is submitted and his parent is this job, we register the task on its parent
	volatile unsigned long nb_subtasks;
	struct splitter_bound_task_pool *subtasks_pool[NMAX_RECTYPE];
	unsigned long nb_subtasks_pool[NMAX_RECTYPE];
	unsigned long n_split; // number of tasks of this type which have been split
	unsigned level;
	char name[128];
};

static struct splitter_bound_task_pool *rec_task_pool[NMAX_LEVEL][NMAX_RECTYPE];
static unsigned splitter_task_max_ind = 0;
//int _starpu_bound_recording;

/* Whether we will include it in the computation */
static int good_job(struct _starpu_job *j)
{
	/* No codelet, nothing to measure */
	if (j->exclude_from_dag)
		return 0;
	if (!j->task->cl)
		return 0;
	/* No performance model, no time duration estimation */
	if (!j->task->cl->model)
		return 0;
	/* Only support history based */
	if (j->task->cl->model->type != STARPU_HISTORY_BASED
	    && j->task->cl->model->type != STARPU_NL_REGRESSION_BASED)
		return 0;
	return 1;
}

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

struct splitter_bound_task_pool *__starpu_splitter_bound_create_task(struct _starpu_job *j)
{
	struct splitter_bound_task_pool *tp;
	_STARPU_MALLOC(tp, sizeof(*tp));
	tp->cl = j->task->cl;
	tp->footprint = j->footprint;
	tp->n = 0;
	tp->n_split = 0;
	tp->level = j->recursive.level;
	tp->nb_subtasks = 0;
	tp->ref_pjob = NULL;
	tp->handle_sizes = 0;
	int nbuf = STARPU_TASK_GET_NBUFFERS(j->task);
	int i = 0;
	sprintf(tp->name, "%s", j->task->name);
	for (i=0; i < nbuf; i++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(j->task, i);
		tp->handle_sizes += _starpu_data_get_size(handle);
	}
	// now we should insert the task on the task pool
	// does it already has an index ?
	unsigned cur_ind = j->task->cl->recursive_task_splitter_register_ind;
	if (cur_ind == 0)
	{
		splitter_task_max_ind += 1;
		j->task->cl->recursive_task_splitter_register_ind = cur_ind = splitter_task_max_ind; // we add 1 before, because when cur_ind is 0, there is no ind.
	}
	STARPU_ASSERT(!rec_task_pool[j->recursive.level][cur_ind-1]);
	rec_task_pool[j->recursive.level][cur_ind-1] = tp;
	return tp;
}

struct splitter_bound_task_pool *__starpu_splitter_get_task(struct _starpu_job *j)
{
	if (j->task->cl->recursive_task_splitter_register_ind == 0)
		return NULL;
	return rec_task_pool[j->recursive.level][j->task->cl->recursive_task_splitter_register_ind-1];
}

unsigned long _starpu_splitter_bound_get_nb_split(struct _starpu_job *j)
{
	if (STARPU_LIKELY(!_starpu_bound_recording))
		return 0;

	if (!good_job(j))
		return 0;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return 0;
	}
	struct splitter_bound_task_pool *tp;

	_starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, NULL, 0, j);

	tp = __starpu_splitter_get_task(j);

		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	if (!tp)
	{
		return 0;
	}
	return tp->n_split;
}

unsigned long _starpu_splitter_bound_get_nb_nsplit(struct _starpu_job *j)
{
	if (STARPU_LIKELY(!_starpu_bound_recording))
		return 0;

	if (!good_job(j))
		return 0;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return 0;
	}
	struct splitter_bound_task_pool *tp;

	_starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, NULL, 0, j);

	tp = __starpu_splitter_get_task(j);

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	if (!tp)
	{
		return 0;
	}
	return tp->n;
}

/* Function used to say to glpk a task has been split for computing area bounds for the splitter*/
void _starpu_splitter_bound_record_split(struct _starpu_job *j)
{
	if (STARPU_LIKELY(!_starpu_bound_recording))
		return;

	if (!good_job(j))
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	{
		struct splitter_bound_task_pool *tp;

		_starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, NULL, 0, j);

		tp = __starpu_splitter_get_task(j);

		if (!tp)
		{
			tp = __starpu_splitter_bound_create_task(j);
		}
		(void) STARPU_ATOMIC_ADD(&tp->n_split, 1);
		/* now looking at parent if exists */
		if (j->recursive.parent_task != NULL)
		{
			struct _starpu_job *pjob = _starpu_get_job_associated_to_task(j->recursive.parent_task);
			_starpu_compute_buffers_footprint(pjob->task->cl ? pjob->task->cl->model:NULL, NULL, 0, pjob);
			struct splitter_bound_task_pool *ptp = __starpu_splitter_get_task(pjob);
			STARPU_ASSERT_MSG(pjob, "Registering subtask without registered parent\n");
			if(ptp->ref_pjob == pjob || ptp->ref_pjob == NULL)
			{
				ptp->ref_pjob = pjob;
				unsigned ind_curtask;
				for (ind_curtask=0; ind_curtask < ptp->nb_subtasks; ind_curtask++)
				{
					if (ptp->subtasks_pool[ind_curtask]->cl == tp->cl && ptp->subtasks_pool[ind_curtask]->footprint == tp->footprint)
						break;
				}
				if (ind_curtask >= ptp->nb_subtasks)
				{
					ptp->subtasks_pool[ptp->nb_subtasks] = tp;
					ptp->nb_subtasks_pool[ptp->nb_subtasks] = 0;
					ind_curtask = ptp->nb_subtasks;
					ptp->nb_subtasks ++;
				}
				ptp->nb_subtasks_pool[ind_curtask] ++;
			}
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/* Function used to say to glpk a task is ready for computing area bounds for the splitter*/
void _starpu_splitter_bound_record(struct _starpu_job *j)
{
	if (STARPU_LIKELY(!_starpu_bound_recording))
		return;

	if (!good_job(j))
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	{
		struct splitter_bound_task_pool *tp = __starpu_splitter_get_task(j);
		_starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, NULL, 0, j);
		if (!tp)
		{
			tp = __starpu_splitter_bound_create_task(j);
		}
		/* One more task of this kind */
		(void) STARPU_ATOMIC_ADD(&tp->n, 1);
		/* now looking at parent if exists */
		if (j->recursive.parent_task != NULL)
		{
			struct _starpu_job *pjob = _starpu_get_job_associated_to_task(j->recursive.parent_task);
			_starpu_compute_buffers_footprint(pjob->task->cl ? pjob->task->cl->model:NULL, NULL, 0, pjob);
			struct splitter_bound_task_pool *ptp = __starpu_splitter_get_task(pjob);
			STARPU_ASSERT_MSG(pjob, "Registering subtask without registered parent\n");
			if(ptp->ref_pjob == pjob || ptp->ref_pjob == NULL)
			{
				ptp->ref_pjob = pjob;
				unsigned ind_curtask;
				for (ind_curtask=0; ind_curtask < ptp->nb_subtasks; ind_curtask++)
				{
					if (ptp->subtasks_pool[ind_curtask]->cl == tp->cl && ptp->subtasks_pool[ind_curtask]->footprint == tp->footprint)
						break;
				}
				if (ind_curtask >= ptp->nb_subtasks)
				{
					ptp->subtasks_pool[ptp->nb_subtasks] = tp;
					ptp->nb_subtasks_pool[ptp->nb_subtasks] = 0;
					ind_curtask = ptp->nb_subtasks;
					ptp->nb_subtasks ++;
				}
				ptp->nb_subtasks_pool[ind_curtask] ++;
			}
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/* Function used to say to glpk a task is ended for computing area bounds for the splitter*/
void _starpu_splitter_bound_delete(struct _starpu_job *j)
{
	if (STARPU_LIKELY(!_starpu_bound_recording))
		return;

	if (!good_job(j))
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}
	{
		struct splitter_bound_task_pool *tp = __starpu_splitter_get_task(j);

		_starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, NULL, 0, j);
		STARPU_ASSERT_MSG(tp, "We delete a task which is not registered\n");
		/* One less task of this kind */
		(void) STARPU_ATOMIC_ADD(&tp->n, -1);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

void _starpu_splitter_bound_delete_split(struct _starpu_job *j)
{
	if (STARPU_LIKELY(!_starpu_bound_recording))
		return;

	if (!good_job(j))
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}
	{
		struct splitter_bound_task_pool *tp = __starpu_splitter_get_task(j);

		_starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, NULL, 0, j);

		STARPU_ASSERT_MSG(tp, "We delete a task which is not registered\n");
		/* One less task of this kind */
		(void) STARPU_ATOMIC_ADD(&tp->n_split, -1);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

}

#ifdef STARPU_HAVE_GLPK_H

unsigned _starpu_splitter_get_max_level_by_type(enum starpu_worker_archtype resource)
{
	if (resource == STARPU_CPU_WORKER)
	{
		return 4;
	}
	return 2; // CUDA_WORKER
}

unsigned _starpu_splitter_get_min_level_by_type(enum starpu_worker_archtype resource)
{
	if (resource == STARPU_CPU_WORKER)
	{
		return 2;
	}
	return 0; // CUDA_WORKER
}

static double cpu_part = 0.; // the needed part of exec_time for cpu
static double cuda_part = 0.; // the needed part of exec_time for cuda
static double cpu_task_factor = 0.; // the factor of the task number for cpu
static double cuda_task_factor = 0.; // the factor of the task number for cuda

void _starpu_splitter_bound_start()
{
	cpu_part = starpu_getenv_float_default("STARPU_RECURSIVE_TASKS_SPLITTER_BOUND_CPU_PART", 1.);
	cuda_part = starpu_getenv_float_default("STARPU_RECURSIVE_TASKS_SPLITTER_BOUND_CUDA_PART", 1.);
	cpu_task_factor = starpu_getenv_float_default("STARPU_RECURSIVE_TASKS_SPLITTER_BOUND_CPU_TASK_FACTOR", 1.);
	cuda_task_factor = starpu_getenv_float_default("STARPU_RECURSIVE_TASKS_SPLITTER_BOUND_CUDA_TASK_FACTOR", 4.);

	fprintf(stderr, "%lf %lf %lf %lf\n", cpu_part, cuda_part, cpu_task_factor, cuda_task_factor);
}

#define STARPU_MAX_REC_TASK_TYPE 48
/* function used to actualize the splitter to compute the ratio of split task*/
void _starpu_splitter_bound_calculate()
{
	glp_term_out(GLP_OFF);

	// for now, we suppose that there is no type of task that appear when splitting a task at level 0
	unsigned max_level = 0;
	unsigned r, l, i, j;

	max_level = NMAX_LEVEL;
	// We disssociate the tasks for the CPU from the one for the CPUs
	// For the same codelet, big ones are for GPUs while smallest are for CPU
	// For now, we do not take into account when having more than 2 recursivity level
	unsigned nb_cpus = starpu_cpu_worker_get_count(), nb_cuda = starpu_cuda_worker_get_count();
	unsigned nb_resources = (nb_cpus != 0) + (nb_cuda != 0);
	unsigned nb_level = max_level;
	unsigned nb_task_used = splitter_task_max_ind;
	glp_prob *lp;
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical cut");
	glp_set_obj_dir(lp, GLP_MIN);
	glp_set_obj_name(lp, "total execution time");

	unsigned real_tasks = 0;
	unsigned t_dif = 0;
	glp_add_rows(lp, 2*nb_resources + (nb_task_used*nb_level));
	// We have :
	// 			- One constraint for each resource, to constraint execution time on the resources
	// 			- One constraint for each resource, to constraint the number of task on each resource
	// 			- On constraint for each task and level, to ensure we have the good number of task split for each level and type

	glp_add_cols(lp, (1+nb_resources)* nb_task_used*nb_level + 1); // nb_resources col for each type of task and each size; and 1 for the exec time
	// We have :
	// 			- One variable for each type of task, level and resource, representing the number of task of this type and this level that are not split
	// 			- One variable for each type of task and level, representing the number of task of this type and this level that are split
	// 			- One variable representing the execution time
	int *ia = calloc((2*nb_resources + nb_task_used*nb_level)*( (1+nb_resources)*nb_level*nb_task_used + 1)+1, sizeof(int));
	int *ja = calloc((2*nb_resources + nb_task_used*nb_level)*( (1+nb_resources)*nb_level*nb_task_used + 1)+1, sizeof(int));
	double *ar = calloc((2*nb_resources + nb_task_used*nb_level)*( (1+nb_resources)*nb_level*nb_task_used + 1)+1, sizeof(double));

	struct starpu_perfmodel_arch *arch_cpu = starpu_worker_get_perf_archtype(starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), STARPU_NMAX_SCHED_CTXS);
	struct starpu_perfmodel_arch *arch_gpu = nb_cuda > 0 ? starpu_worker_get_perf_archtype(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), STARPU_NMAX_SCHED_CTXS) : NULL;

	// Firstly, we set the row number and the column number
	for (i = 0; i < 2*nb_resources + nb_task_used*nb_level; i++)
	{
		for (j = 0; j < (1+nb_resources)*nb_task_used*nb_level + 1; j++)
		{
			// setting first the contribution for the lines of each resources
			ia[i*((1+nb_resources)*nb_level*nb_task_used + 1) + j + 1] = i + 1;
			ja[i*((1+nb_resources)*nb_level*nb_task_used + 1) + j + 1] = j + 1;
		}
	}

	// at this point, ia and ja are correcly set, and ar is set to 0. everywhere

	unsigned splitter_task_ind;
	for (splitter_task_ind=0; splitter_task_ind < nb_task_used; splitter_task_ind++)
	{
		// The first step is to set correctly the rows of the resource constraint

		for (l=0; l < max_level; l++)
		{ // On this loop, we say that for each task that is dec, it takes 20 us of CPU time
			ar[
				// No jump of line, we add time at CPU constraint
				(1+nb_resources) * splitter_task_ind*nb_level + // we jump all the first tasks, we multiply by 2 because we have a number for dec and a number for ndec
				(1+nb_resources) * l + // jump the first levels
				+ 1 // jump of GLPK
				] = 100; // For each task split, it takes 20 us of CPU time to be managed
		}
		for (r=0; r < nb_resources; r++)
		{
			// r = 0 -> CPU, r = 1 -> CUDA
			//		unsigned min_level_resource = _starpu_splitter_get_min_level_by_type(r),
			//							max_level_resource = _starpu_splitter_get_max_level_by_type(r);
			unsigned min_level_resource = 0;
			unsigned max_level_resource = (r == 0 || nb_cuda > 0) ? max_level : 0;
			for (l=min_level_resource; l<max_level_resource; l++)
			{
				double time = 0.;
				unsigned task_exist = 0;
				if (rec_task_pool[l][splitter_task_ind] != NULL)
				{
					struct _starpu_job job =
					{
						.footprint = rec_task_pool[l][splitter_task_ind]->footprint,
						.footprint_is_computed = 1,
					};
					time = _starpu_history_based_job_expected_perf(rec_task_pool[l][splitter_task_ind]->cl->model, r == 0 ? arch_cpu : arch_gpu, &job, job.nimpl);
					if (!isnan(time))
					{
						task_exist = 1;
					}
					else
					{
						time = 0.;
					}
					if (r == 0)
						time = (time + 5);
					else
						time += 10; // this is GPU management
				}

				ar[
					r * ((1+nb_resources)*nb_task_used*nb_level + 1) + // we jump r lines
					(1+nb_resources) * splitter_task_ind*nb_level + // we jump all the first tasks, we multiply by 2 because we have a number for dec and a number for ndec
					(1+nb_resources) * l + // jump the first levels
					1 + r // first we have nb_dec and after nb_ndec for each resource
					+ 1 // jump of GLPK
					] = time; // this initialize the value for the resource constraint
				ar[
					// No jump of line, we add time at CPU constraint
					(1+nb_resources) * splitter_task_ind*nb_level + // we jump all the first tasks, we multiply by 2 because we have a number for dec and a number for ndec
					(1+nb_resources) * l + // jump the first levels
					1+r +
					+ 1 // jump of GLPK
					] += 10; // each task has 20us time of handling by a CPU

				ar[
					(r+nb_resources) * ((1+nb_resources)*nb_task_used*nb_level + 1) + // we jump nb_resources+r lines to go to the constraint of task number by resources
					(1+nb_resources) * splitter_task_ind*nb_level + // we jump all the first tasks, we multiply by 2 because we have a number for dec and a number for ndec
					(1+nb_resources) * l + // jump the first levels
					1 +r// first we have nb_dec and after nb_ndec
					+ 1 // jump of GLPK
					] =  task_exist; // this initialize the constraint corresponding to the sum of the number of task we would like for each resource
			}
		}
		// now we initialize to obtain the good number of task for each level
		// we start our initialization by the number of split task for each level

		for (l=0; l<nb_level; l++)
		{
			if (l == 0 && rec_task_pool[l][splitter_task_ind] )
			{
				if (rec_task_pool[l][splitter_task_ind]->n + rec_task_pool[l][splitter_task_ind]->n_split > 0 )
				{
					real_tasks += rec_task_pool[l][splitter_task_ind]->n + rec_task_pool[l][splitter_task_ind]->n_split;
					t_dif ++;
				}
			}

			glp_set_row_bnds(lp, 2*nb_resources + splitter_task_ind*nb_level + l + 1, GLP_LO, rec_task_pool[l][splitter_task_ind] ? rec_task_pool[l][splitter_task_ind]->n : 0., 0.); // The number of tasks of this type is nb
			char name[256];
			sprintf(name, "task %s level %u", rec_task_pool[l][splitter_task_ind] ? rec_task_pool[l][splitter_task_ind]->name : "nexist", l);
			glp_set_row_name(lp, 2*nb_resources + splitter_task_ind*nb_level + l + 1, name);

			glp_set_col_bnds(lp, (1+nb_resources)*splitter_task_ind*nb_level + (1+nb_resources)*l + 1, GLP_LO, 0.0, 0.0); // nb tasks split is >= 0

			sprintf(name, "task %s level %u split", rec_task_pool[l][splitter_task_ind] ? rec_task_pool[l][splitter_task_ind]->name : "nexist", l);
			glp_set_col_name(lp, (1+nb_resources)*splitter_task_ind*nb_level + (1+nb_resources)*l + 1, name);
			if (rec_task_pool[l][splitter_task_ind] == NULL)
				continue;
			ar[
				(2*nb_resources + splitter_task_ind*nb_level+l) * ((1+nb_resources)*nb_task_used*nb_level + 1) + // jump the first lines : the ones corresponding to resource time, the ones corresponding to ressource ntasks, and the ones corresponding to the first tasks, and also the ones corresponding to the first levels' task
				(1+nb_resources)*splitter_task_ind*nb_level + // jump the first task
				(1+nb_resources)*l //  the first levels
				+1 // glpk jump
				] = rec_task_pool[l][splitter_task_ind]->nb_subtasks > 0; // this set the number of task of this level that are split
			// now we will set this line for our children tasks, if exists
			unsigned ii;
			for (ii = 0; ii < rec_task_pool[l][splitter_task_ind]->nb_subtasks; ii++)
			{
				struct splitter_bound_task_pool *little_task = rec_task_pool[l][splitter_task_ind]->subtasks_pool[ii];
				unsigned nb_subtasks = rec_task_pool[l][splitter_task_ind]->nb_subtasks_pool[ii];
				ar[
					(2*nb_resources + (little_task->cl->recursive_task_splitter_register_ind-1)*nb_level + l + 1) * ((1+nb_resources)*nb_task_used*nb_level + 1) + // jump the first lines : the ones corresponding to resource time, the ones corresponding to ressource ntasks, the ones corresponding to the task before our child, and also the ones corresponding to the levels before our child level
					(1+nb_resources)*splitter_task_ind*nb_level + // jump the tasks before the big task
					(1+nb_resources)*l + //  the first levels
					+1 // glpk jump
					] = nb_subtasks*-1.; // We want to utilize the nb_split of this task this set the number of task of this level that are not split
			}
			for (r=0; r < nb_resources; r++)
			{
				glp_set_col_bnds(lp, (1+nb_resources)*splitter_task_ind*nb_level + (1+nb_resources)*l +1+r + 1, GLP_LO, 0.0, 0.0); // nb tasks nsplit is >= 0
				sprintf(name, "task %s level %u nsplit resource %d", rec_task_pool[l][splitter_task_ind] ? rec_task_pool[l][splitter_task_ind]->name : "nexist", l, r);
				glp_set_col_name(lp, (1+nb_resources)*splitter_task_ind*nb_level + (1+nb_resources)*l +1+r +1, name);

				ar[
					(2*nb_resources + splitter_task_ind*nb_level + l) * ((1+nb_resources)*nb_task_used*nb_level + 1) + // jump the first lines : the ones corresponding to resource time, the ones corresponding to ressource ntasks, and the ones corresponding to the first tasks, and also the ones corresponding to the first levels' task
					(1+nb_resources)*splitter_task_ind*nb_level + // jump the first task
					(1+nb_resources)*l + //  the first levels
					1 + r// the task that are split
					+1 // glpk jump
					] = 1.;
				/*ar[(r+nb_resources) * ((1+nb_resources)*nb_task_used*nb_level + 1) + // we jump nb_resources+r lines to go to the constraint of task number by resources
				  (1+nb_resources) * splitter_task_ind*nb_level + // we jump all the first tasks, we multiply by 2 because we have a number for dec and a number for ndec
				  (1+nb_resources) * l + // jump the first levels
				  1 + r// first we have nb_dec and after nb_ndec
				  + 1 // jump of GLPK
				  ]; // If a task cannot be executed on one or the other device, then it cannot be not split
				*/}
		}
	}
	// now we set for each resource
	glp_set_col_bnds(lp, (1+nb_resources)*nb_task_used*nb_level + 1, GLP_LO, 0.0, 0.0); // bigger than 0
	glp_set_col_name(lp, (1+nb_resources)*nb_task_used*nb_level + 1, "t_exec");
	// now we have to set the number of resources for t_exec
	for (r=0; r<nb_resources; r++)
	{
		ar[
			(r) * ((1+nb_resources)*nb_task_used*nb_level + 1) + // jump the first lines : the ones corresponding to other resources timee
			(1+nb_resources)*nb_task_used*nb_level // jump the tasks
			+1 // GLPK jump
			] = -1. * (r==0 ? cpu_part*nb_cpus : cuda_part*nb_cuda);

		// don't forget bounds !
		glp_set_row_bnds(lp, r+1, GLP_UP, 0., 0.); // The number of tasks of this type is nb
		glp_set_row_bnds(lp, nb_resources + r+1, GLP_LO, r==0 ? cpu_task_factor*nb_cpus : cuda_task_factor*nb_cuda, 0.); // The number of tasks of this type is nb
		glp_set_row_name(lp, r+1, r == 0 ? "time CPU" : "time CUDA");
		glp_set_row_name(lp, nb_resources + r+1, r == 0 ? "nb tasks CPU" : "nb tasks CUDA");
	}

	/*for (i = 0; i < 2*nb_resources + nb_task_used*nb_level; i++)
	{
		for (j = 0; j < (1+nb_resources)*nb_task_used*nb_level + 1; j++)
		{
			fprintf(stderr, "%.2lf ", ar[i*((1+nb_resources)*nb_level*nb_task_used + 1) + j + 1]);
		}
		fprintf(stderr, "\n");
	}*/

//	glp_smcp parm = {
//	GLP_MSG_ALL, GLP_DUAL, GLP_PT_PSE, GLP_RT_HAR, 1e-7, 1e-7, 1e-9, -DBL_MAX, DBL_MAX, INT_MAX, 50, 0, GLP_ON
//	};

	// we would like to minimize the execution time
	glp_set_obj_coef(lp, 2*nb_task_used*nb_level+1, 1.);
	//fprintf(stderr, "write problem with %d tasks\n", nb_task_used);
	glp_load_matrix(lp, ((1+nb_resources)*nb_level*nb_task_used +1)*(2*nb_resources + nb_task_used*nb_level) , ia, ja, ar);
	//char name[256];
	//sprintf(name, "problem_%u_tasks_%u", real_tasks, t_dif);
	//glp_write_lp(lp, NULL, name);
	//struct timespec ts, ts2;
	//_starpu_clock_gettime(&ts);
	//fprintf(stderr, "solve : %d\n", glp_simplex(lp, NULL));
	glp_simplex(lp, NULL);
	//sprintf(name, "sol_%d_tasks", real_tasks);
	//glp_print_sol(lp, name);
	//_starpu_clock_gettime(&ts2);
	// There, we have one ratio for each type of task.
	// We will put it on the codelet
	//double nus = starpu_timing_timespec_delay_us(&ts, &ts2);
	//fprintf(stderr, "%lf\n", nus);
	_splitter_reinit_cache_entry();

	for (splitter_task_ind=0; splitter_task_ind < nb_task_used; splitter_task_ind++)
	{
		for (l=0; l < nb_level; l++)
		{
			double ntasks_split = glp_get_col_prim(lp, (1+nb_resources)*nb_level*splitter_task_ind + (1+nb_resources)*l +1);
			double ntasks_nsplit = 0;
			for (r=0; r < nb_resources; r++)
				ntasks_nsplit += glp_get_col_prim(lp, (1+nb_resources)*nb_level*splitter_task_ind + (1+nb_resources)*l +1 +1+r);
			double ntasks = ntasks_split + ntasks_nsplit;
			//if (ntasks > 0)
			{
				double ratio = ntasks_split/ntasks;
				//	if (l <= 1 )
				//fprintf(stderr, "(%d) task %s level %d has ratio %lf : nsplit = %lf : nnsplit = %lf\n", real_tasks,  !rec_task_pool[l][splitter_task_ind] ? NULL : rec_task_pool[l][splitter_task_ind]->name, l, ratio, ntasks_split, ntasks_nsplit);
				if (ntasks>0)
					_splitter_actualize_ratio_to_split(rec_task_pool[l][splitter_task_ind]->cl, ratio, l);
				else if (rec_task_pool[l][splitter_task_ind])
					_splitter_actualize_ratio_to_split(rec_task_pool[l][splitter_task_ind]->cl, 0., l);
			}
		}
	}
	//if (real_tasks > 3)
	//	abort();
	//fprintf(stderr, "solve with %d tasks\n", real_tasks);

	glp_delete_prob(lp);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	free(ia);
	free(ja);
	free(ar);
}
#endif /* STARPU_HAVE_GLPK_H */

#endif // STARPU_RECURSIVE_TASKS
