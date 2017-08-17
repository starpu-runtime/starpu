/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2017  Université de Bordeaux
 * Copyright (C) 2017  Erwan Leria
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
 * This reads a tasks.rec file and replays the recorded task graph.
 * Currently, this is done for simgrid
 */

#include <starpu.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>

#include <common/uthash.h>
#include <starpu_scheduler.h>

#define NMAX_DEPENDENCIES 16


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Declarations of global variables, structures, pointers, ... *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

typedef unsigned long jobid_t;

static char *name = NULL;
static char *model = NULL;
static jobid_t jobid;
static jobid_t *dependson;
static unsigned int submitorder;
static starpu_tag_t tag;
static int workerid;
static uint32_t footprint;
static double flops;

static double startTime; //start time (The instant when the task starts)
static double endTime; //end time (The instant when the task ends)

static int iteration = -1;

static starpu_data_handle_t handles[STARPU_NMAXBUFS];
static enum starpu_data_access_mode modes[STARPU_NMAXBUFS];

/* Use the following arrays when the number of data is greater than STARPU_NMAXBUFS */

starpu_data_handle_t * handles_ptr;
enum starpu_data_access_mode * modes_ptr;
size_t * sizes_set;

static size_t dependson_size;
static size_t ndependson;

static int nb_parameters = 0; /* Nummber of parameters */
static int alloc_mode; /* If alloc_mode value is 1, then the handles are stored in dyn_handles, else they are in handles */

static unsigned int priority = 0;

char * reg_signal = NULL; /* The register signal (0 or 1 coded on 8 bit) it is used to know which handle of the task has to be registered in StarPu */

static int device;

/* Record all tasks, hashed by jobid. */
static struct task
{
	UT_hash_handle hh;
	jobid_t jobid;
        int iteration;
	struct starpu_task task;
} *tasks;

/* Record handles */
static struct handle
{
	UT_hash_handle hh;
	starpu_data_handle_t mem_ptr; /* This value should be the registered handle */
	starpu_data_handle_t handle; /* The key is the original value of the handle in the file */
} * handles_hash;

/* Record models */

static struct perfmodel
{
	UT_hash_handle hh;
	struct starpu_perfmodel perfmodel;
	char * model_name;
} * model_hash;

/* Record a dependent task by its jobid and the jobids of its dependencies */
struct s_dep
{
	jobid_t job_id;
	jobid_t deps_jobid[NMAX_DEPENDENCIES]; /* That array has to contain the jobids of the dependencies, notice that the number of dependcies is limited to 16, modify NMAX_DEPENDENCIES at your convenience */
	size_t ndependson;
};

/* Declaration of an AoS (array of structures) */
struct s_dep ** jobidDeps;
size_t jobidDeps_size;
static size_t ntask = 0; /* This is the number of dependent task (le nombre de tâches dépendantes) */

/* Settings for the perfmodel */
struct task_arg
{
	uint32_t footprint;
	double perf[];
};

uint32_t get_footprint(struct starpu_task * task)
{
	return ((struct task_arg*) (task->cl_arg))->footprint;
}

double arch_cost_function(struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned nimpl)
{
	device = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);

	/* Then, get the pointer to the value of the expected time */
	double * val = (double *) ((struct task_arg *) (task->cl_arg))->perf+device;

	if (!(*val == 0 || isnan(*val)))
		return *val;

	fprintf(stderr, "[starpu] Error, expected_time is 0 or lower (replay.c line : %d)", __LINE__- 6);

	return 0.0;
}

/* End of settings */

void dumb_kernel(void) {}

/* [CODELET] Initialization of an unique codelet for all the tasks*/
static int can_execute(unsigned worker_id, struct starpu_task *task, unsigned nimpl)
{
	struct starpu_perfmodel_arch * arch = starpu_worker_get_perf_archtype(worker_id, STARPU_NMAX_SCHED_CTXS);
	double expected_time = ((struct task_arg *) (task->cl_arg))->perf[(starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices))];
	if (!(expected_time == 0 || isnan(expected_time)))
	{
		return 1;
	}

	return 0;
}

static struct starpu_perfmodel myperfmodel =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = arch_cost_function,
	.footprint = get_footprint,
};

static struct starpu_codelet cl =
{
	.cpu_funcs = { (void*) 1 },
	.cpu_funcs_name = { "dumb_kernel" },
	.cuda_funcs = { (void*) 1 },
	.opencl_funcs = { (void*) 1 },
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
	.can_execute = can_execute,
	.model = &myperfmodel,
};

/* * * * * * * * * * * * * *
* * * * * Functions * * * * *
* * * * * * * * * * * * * * */

/* Initializing an array with 0 */
void array_init(unsigned long * arr, int size)
{
	unsigned i;
	for (i = 0 ; i < size ; i++)
	{
		arr[i] = 0;
	}
}

/* Initializing s_dep structure */
struct s_dep * s_dep_init(struct s_dep * sd, jobid_t job_id)
{
	sd = malloc(sizeof(*sd));
	sd->job_id = jobid;
	array_init((unsigned long *) sd->deps_jobid, 16);
	sd->ndependson = 0;

	return sd;
}

/* Remove s_dep structure */
void s_dep_remove(struct s_dep * sd)
{
	free(sd);
}

/* Array duplication */
static void array_dup(unsigned long * in_arr, unsigned long * out_arr, int size)
{
	int i;

	for (i = 0 ; i < size ; i++)
	{
		out_arr[i] = in_arr[i];
	}
}

/* The following function checks if the program has to use static or dynamic arrays*/
static int set_alloc_mode(int total_parameters)
{
	return (total_parameters <= STARPU_NMAXBUFS);
}

/* According to the allocation mode, modify handles_ptr and modes_ptr in static or dynamic */
static void arrays_managing(int mode)
{
	if (mode)
	{
		handles_ptr = &handles[0];
		modes_ptr = &modes[0];
	}
	else
	{
		handles_ptr =  malloc(sizeof(*handles_ptr) * nb_parameters);
		modes_ptr = malloc(sizeof(*modes_ptr) * nb_parameters);

	}
}

/* Check if a handle hasn't been registered yet */
static void variable_data_register_check(size_t * array_of_size, int nb_handles)
{
	int h;

	for (h = 0 ; h < nb_handles ; h++)
	{
		if(reg_signal[h]) /* Get the register signal, and if it's 1 do ... */
		{
			struct handle * strhandle_tmp;

			/* Find the key that was stored in &handles_ptr[h] */

			HASH_FIND(hh, handles_hash, handles_ptr+h, sizeof(handles_ptr[h]), strhandle_tmp);

			starpu_variable_data_register(handles_ptr+h, STARPU_MAIN_RAM, (uintptr_t) 1, array_of_size[h]);

			strhandle_tmp->mem_ptr = handles_ptr[h];
		}
	}
}

void reset(void)
{
	if (name != NULL)
	{
		free(name);
		name = NULL;
	}

	if (model != NULL)
	{
		free(model);
		model = NULL;
	}

	if (sizes_set != NULL)
	{
		free(sizes_set);
		sizes_set = NULL;
	}

	if reg_signal != NULL)
	{
		free(reg_signal);
		reg_signal = NULL;
	}

	jobid = 0;
	submitorder = 0;
	ndependson = 0;
	tag = -1;
	workerid = -1;
	footprint = 0;
	startTime = 0.0;
	endTime = 0.0;
	iteration = -1;
	nb_parameters = 0;
	alloc_mode = 1;
}

/* Functions that submit all the tasks (used when the program reaches EOF) */
int submit_tasks(void)
{
	/* Add dependencies */
	int j;

	for(j = 0; j < ntask ; j++)
	{
		struct task * currentTask;

		/* Looking for the task associate to the jobid of the jth element of jobidDeps */
		HASH_FIND(hh, tasks, &jobidDeps[j]->job_id, sizeof(jobid), currentTask);

		if (jobidDeps[j]->ndependson > 0)
		{
			struct starpu_task * taskdeps[jobidDeps[j]->ndependson];
			unsigned i;

			for (i = 0; i < jobidDeps[j]->ndependson; i++)
			{
				struct task * taskdep;

				/*  Get the ith jobid of deps_jobid */
				HASH_FIND(hh, tasks, &jobidDeps[j]->deps_jobid[i], sizeof(jobid), taskdep);

				assert(taskdep);
				taskdeps[i] = &taskdep->task;
			}

			starpu_task_declare_deps_array(&currentTask->task, jobidDeps[j]->ndependson, taskdeps);
		}

		if (!(currentTask->iteration == -1))
			starpu_iteration_push(currentTask->iteration);

		int ret_val = starpu_task_submit(&currentTask->task);

		if (!(currentTask->iteration == -1))
			starpu_iteration_pop();

		//printf("name : %s \n", currentTask->task.name);

		printf("submitting task %s (%lu, %llu)\n", currentTask->task.name?currentTask->task.name:"anonymous", jobidDeps[j]->job_id, (unsigned long long) currentTask->task.tag_id /* tag*/);

		if (ret_val != 0)
			return -1;
	}

	return 1;
}


/* * * * * * * * * * * * * * * */
/* * * * * * MAIN * * * * * * */
/* * * * * * * * * * * * * * */

int main(int argc, char **argv)
{
	starpu_data_set_default_sequential_consistency_flag(0);

	FILE * rec;
	char * s;
	size_t s_allocated = 128;
	s = malloc(s_allocated);

	dependson_size = NMAX_DEPENDENCIES;
	jobidDeps_size = 512;

	dependson =  malloc(dependson_size * sizeof (* dependson));
	jobidDeps = malloc(jobidDeps_size * sizeof(* jobidDeps));

	alloc_mode = 1;

	if (argc <= 1)
	{
		fprintf(stderr,"Usage: %s tasks.rec [ordo.rec]\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	rec = fopen(argv[1], "r");
	if (!rec)
	{
		fprintf(stderr,"unable to open file %s: %s\n", argv[1], strerror(errno));
		exit(EXIT_FAILURE);
	}

	int ret = starpu_init(NULL);
	if (ret == -ENODEV) goto enodev;

	/* Read line by line, and on empty line submit the task with the accumulated information */
	reset();

	while(1)
	{
		char *ln;

		if (!fgets(s, s_allocated, rec))
		{
			int submitted = submit_tasks();

			if (submitted == -1)
			{
				goto enodev;
			}

			goto eof;
		}

		while (!(ln = strchr(s, '\n')))
		{
			/* fprintf(stderr,"buffer size %d too small, doubling it\n", s_allocated); */
			s = realloc(s, s_allocated * 2);

			if (!fgets(s + s_allocated-1, s_allocated+1, rec))
			{
				int submitted = submit_tasks();

				if (submitted == -1)
				{
					goto enodev;
				}

				goto eof;
			}

			s_allocated *= 2;
		}

		if (ln == s)
		{
			/* Empty line, do task */
			struct task * task = malloc(sizeof(*task));

			task->jobid = jobid;
			task->iteration = iteration;
			starpu_task_init(&task->task);

			if (name != NULL)
				task->task.name = strdup(name);
			task->task.submit_order = submitorder;

			/* Check workerid */
			if (workerid >= 0)
			{
				task->task.cl = &cl;
				task->task.workerid = workerid;

				if (alloc_mode)
				{
					/* Duplicating the handles stored (and registered in the current context) into the task */

					array_dup((unsigned long *) modes_ptr, (unsigned long *) task->task.modes, nb_parameters);
					array_dup((unsigned long *) modes_ptr, (unsigned long *) task->task.cl->modes, nb_parameters);
					variable_data_register_check(sizes_set, nb_parameters);
					array_dup((unsigned long *) handles_ptr, (unsigned long *) task->task.handles, nb_parameters);
				}
				else
				{
					task->task.dyn_modes = modes_ptr;
					task->task.cl->dyn_modes = malloc(sizeof(starpu_data_handle_t) * nb_parameters);
					array_dup((unsigned long *) modes_ptr, (unsigned long *) task->task.cl->dyn_modes, nb_parameters);
					variable_data_register_check(sizes_set, nb_parameters);
					task->task.dyn_handles = handles_ptr;
				}

				task->task.nbuffers = nb_parameters;

				struct perfmodel * realmodel;

				HASH_FIND_STR(model_hash, model, realmodel);

				if (realmodel == NULL)
				{
					int len = strlen(model);
					realmodel =  calloc(1, sizeof(struct perfmodel));

					if (realmodel == NULL)
					{
						fprintf(stderr, "[starpu][Error] Not enough memory for allocation ! (replay.c)");
						exit(EXIT_FAILURE);
					}

					realmodel->model_name = (char *) malloc(sizeof(char) * (len+1));
					realmodel->model_name = strcpy(realmodel->model_name, model);

					starpu_perfmodel_init(&realmodel->perfmodel);

					HASH_ADD_STR(model_hash, model_name, realmodel);

					int error = starpu_perfmodel_load_symbol(model, &realmodel->perfmodel);

					if (error)
					{
						fprintf(stderr, "[starpu][Warning] Error loading perfmodel symbol");
					}

				}

				int narch = starpu_perfmodel_get_narch_combs();
				struct task_arg * arg = malloc(sizeof(struct task_arg) + sizeof(double) * narch);
				arg->footprint = footprint;
				double * perfTime  = arg->perf;
				int i;

				for (i = 0; i < narch ; i++)
				{
					struct starpu_perfmodel_arch * arch = starpu_perfmodel_arch_comb_fetch(i);
					perfTime[i] = starpu_perfmodel_history_based_expected_perf(&realmodel->perfmodel, arch, footprint);
				}

				task->task.cl_arg = arg;
				task->task.flops = flops;

			}

			task->task.cl_arg_size = 0;
			task->task.tag_id = tag;
			task->task.use_tag = 1;

			struct s_dep * sd = s_dep_init(sd, jobid);
			array_dup((unsigned long *) dependson, (unsigned long *) sd->deps_jobid, ndependson);
			sd->ndependson = ndependson;

			if (ntask >= jobidDeps_size)
			{
				jobidDeps_size *= 2;
				jobidDeps = realloc(jobidDeps, jobidDeps_size * sizeof(*jobidDeps));
			}

			jobidDeps[ntask] = sd;
			ntask++;

			// TODO: call applyOrdoRec(task);
			//
			// Tag: 1234
			// Priority: 12
			// Workerid: 0   (-> ExecuteOnSpecificWorker)
			// Workers: 0 1 2
			// DependsOn: 1235
			//
			// PrefetchTag: 1234
			// DependsOn: 1233


			/* Add this task to task hash */
			HASH_ADD(hh, tasks, jobid, sizeof(jobid), task);

			reset();
		}

		/* Record various information */
#define TEST(field) (!strncmp(s, field": ", strlen(field) + 2))

		else if (TEST("Name"))
		{
			*ln = 0;
			name = strdup(s+6);
		}
		else if (TEST("Model"))
		{
			*ln = 0;
			model = strdup(s+7);
		}
		else if (TEST("JobId"))
			jobid = atol(s+7);
		else if(TEST("SubmitOrder"))
			submitorder = atoi(s+13);
		else if (TEST("DependsOn"))
		{
			char *c = s + 11;

			for (ndependson = 0; *c != '\n'; ndependson++)
			{
				if (ndependson >= dependson_size)
				{
					dependson_size *= 2;
					dependson = realloc(dependson, dependson_size * sizeof(*dependson));
				}
				dependson[ndependson] = strtol(c, &c, 10);
			}
		}
		else if (TEST("Tag"))
			tag = strtol(s+5, NULL, 16);
		else if (TEST("WorkerId"))
			workerid = atoi(s+10);
		else if (TEST("Footprint"))
		{
			footprint = strtoul(s+11, NULL, 16);
		}
		else if (TEST("Parameters"))
		{
			/* Parameters line format is PARAM1_PARAM2_(...)PARAMi_(...)PARAMn */
			char * param_str = s + 12;
			unsigned i;
			int count = 0;

			for (i = 0 ; param_str[i] != '\n'; i++)
			{
				if (param_str[i] == '_') /* Checking the number of '_' (underscore), assuming that the file is not corrupted */
				{
					count++;
				}
			}

			nb_parameters = count + 1; /* There is one underscore per paramater execept for the last one, that's why we have to add +1 (dirty programming) */

			alloc_mode = set_alloc_mode(nb_parameters);

			arrays_managing(alloc_mode);

			reg_signal = (char *) calloc(nb_parameters, sizeof(char));
		}
		else if (TEST("Handles"))
		{
			char * buffer = s + 9;
			const char * delim = " ";
			char * token = strtok(buffer, delim);

			while (token != NULL)
			{
				int i;

				for (i = 0 ; i < nb_parameters ; i++)
				{
					struct handle * handles_cell; /* A cell of the hash table for the handles */
					starpu_data_handle_t  handle_value = (starpu_data_handle_t) strtol(token, NULL, 16); /* Get the ith handle on the line */

					HASH_FIND(hh, handles_hash, &handle_value, sizeof(handle_value), handles_cell); /* Find if the handle_value was already registered as a key in the hash table */

					if (handles_cell == NULL)
					{
						handles_cell = malloc(sizeof(*handles_cell));
						handles_cell->handle = handle_value;

						HASH_ADD(hh, handles_hash, handle, sizeof(handle_value), handles_cell); /* If it wasn't, then add it to the hash table */

						handles_ptr[i] = handle_value;

						reg_signal[i] = 1;
					}
					else
					{
						handles_ptr[i] = handles_cell->mem_ptr;
					}

					token = strtok(NULL, delim);
				}
			}
		}
		else if (TEST("Modes"))
		{
			char * buffer = s + 7;
			int mode_i = 0;
			int i = 0;
			const char * delim = " ";
			char * token = strtok(buffer, delim);

			while (token != NULL)
			{
				/* Subject to the names of starpu modes enumerator are not modified */
				if (!strncmp(token, "RW", 2))
				{
					*(modes_ptr+mode_i) = STARPU_RW;
					mode_i++;
					i++;
				}
				else if (!strncmp(token, "R", 1))
				{
					*(modes_ptr+mode_i) = STARPU_R;
					mode_i++;
				}
				else if (!strncmp(token, "W", 1))
				{
					*(modes_ptr+mode_i) = STARPU_W;
					mode_i++;
				}
				/* Other cases produce a warning*/
				else
				{
					fprintf(stderr, "[Warning] A mode is different from R/W (jobid task : %lu)", jobid);
				}
				token = strtok(NULL, delim);
			}
		}
		else if (TEST("Sizes"))
		{
			char *  buffer = s + 7;
			const char * delim = " ";
			char * token = strtok(buffer, delim);
			int k = 0;

			sizes_set = (size_t *) malloc(nb_parameters * sizeof(size_t));

			while (token != NULL)
			{
				sizes_set[k] = strtol(token, NULL, 10);
				token = strtok(NULL, delim);

				k++;
			}
		}
		else if (TEST("StartTime"))
		{
			startTime = strtod(s+11, NULL);
		}
		else if (TEST("EndTime"))
		{
			endTime = strtod(s+9, NULL);
		}
		else if (TEST("GFlop"))
		{
			flops = 1000000000 * strtod(s+7, NULL);
		}
		else if (TEST("Iteration"))
		{
			iteration = (unsigned) strtol(s+11, NULL, 10);
		}
		else if (TEST("Priority"))
		{
			priority = strtol(s + 10, NULL, 10);
		}
		/* ignore */
		//else fprintf(stderr,"warning: unknown '%s' field\n", s);
	}

eof:

	starpu_task_wait_for_all();

	/* FREE allocated memory */

	free(dependson);
	free(s);

	for (alloc_mode  = 0; alloc_mode < ntask ; alloc_mode++)
	{
		s_dep_remove(jobidDeps[alloc_mode]);
	}

	free(jobidDeps);

	/* End of FREE */

	struct handle * handle,* handletmp;
	HASH_ITER(hh, handles_hash, handle, handletmp)
	{
		starpu_data_unregister(handle->mem_ptr);
		HASH_DEL(handles_hash, handle);
		free(handle);
        }

	starpu_shutdown();

	struct perfmodel * model_s, * modeltmp;
	HASH_ITER(hh, model_hash, model_s, modeltmp)
	{
		starpu_perfmodel_unload_model(&model_s->perfmodel);
		HASH_DEL(model_hash, model_s);
		free(model_s->model_name);
		free(model_s);
        }

	struct task * task, *tasktmp;
	HASH_ITER(hh, tasks, task, tasktmp)
	{
		free(task->task.cl_arg);
		free((char*)task->task.name);
		HASH_DEL(tasks, task);
		starpu_task_clean(&task->task);
		free(task);
        }

	return 0;

enodev:
	return 77;
}
