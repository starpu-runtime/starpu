/*SCHED.REC*/

#include <starpu.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <common/uthash.h>
#include <common/utils.h>


#define CPY(src, dst, n) memcpy(dst, src, n * sizeof(*dst))

static unsigned eosw;
static unsigned priority;
static unsigned workerorder;
static unsigned workers[STARPU_NMAXWORKERS];
static unsigned dependson[STARPU_NMAXBUFS];
static unsigned nworkers;
static unsigned nparam;
static unsigned memnode;
static unsigned ndependson;
static unsigned submitorder; /* Also use as prefetchtag */
static unsigned parameters[STARPU_NMAXBUFS];

static unsigned add_to_hash = 0;
static int sched_type = -1;
/* sched_type is called s_type in the structure struct task 
   - If s_type == -1, no task has been added to the structure
   - If s_type == 0, a false task has been added to the structure (at index 1)
   - If s_type == 1, scheduling info has been added into the non-false task (at index 0) of the structure
   - If s_type == 2, a false and a true task have been added into the structure
*/

static struct starpu_codelet cl_prefetch = {
        .where = STARPU_NOWHERE,
        .nbuffers = 1,
        .modes = { STARPU_R },
};

static struct task
{
	UT_hash_handle hh;
	unsigned submitorder;
	int pref_dep;
	unsigned send;
	/* "Prefetch dependence" is the submit order of the dependence for the prefetch, we've chosen to work with only one value, but it can be more (in this case rearrange the code 
			      and add eventually a new field in this structure named npref_dep or something like that) */
	struct starpu_task tasks[2]; /* It seems that we only need 2 slots, one for the scheduling info (stored in a task), and another for the false task */
	struct starpu_task pref_task;
	unsigned dependson[STARPU_NMAXBUFS];
	unsigned ndependson;
	unsigned parameters[STARPU_NMAXBUFS];
	unsigned nparameters;
	unsigned memory_node;
	unsigned s_type;
} *sched_data;

/* TODO : respecter l'ordre de soumission des tâches SubmitOrder */

// TODO: call SchedRecInit


void checkField(char * s)
{
	if (!strncmp(s, "SubmitOrder: ", sizeof("SubmitOrder: ")))
	{
		s = s + sizeof("SubmitOrder: ");
		submitorder = strtol(s, NULL, 16);
		sched_type += 2;
	}

	else if (!strncmp(s, "Priority: ", sizeof("Prioriyty: ")))
	{
		s = s + sizeof("Priority: ");
		priority = strtol(s, NULL, 10);
	}

	else if (!strncmp(s, "ExecuteOnSpecificWorker: ", sizeof("ExecuteOnSpecificWorker: ")))
	{
		eosw = strtol(s, NULL, 10);
	}

	else if (!strncmp(s, "Workers: ", sizeof("Workers: ")))
	{
		s = s + sizeof("Workers: ");
		char * delim = " ";
		char * token = strtok(s, delim);
		int i = 0;
		 
		while (token != NULL)
		{
			int k = strtol(token, NULL, 10);
			workers[k/sizeof(*workers)] |= (1 << (k%(sizeof(*workers))));
			i++;
		}

		nworkers = i;
	}

	else if (!strncmp(s, "DependsOn: ", sizeof("DependsOn: ")))
	{
		/* NOTE : dependsons (in the sched.rec)  should be the submit orders of the dependences, 
		   otherwise it can occur an undefined behaviour
		   (contrary to the tasks.rec where dependences are jobids */
		unsigned i = 0;
		char * delim = " ";
		char * token = strtok(s+sizeof("DependsOn: "), delim);
		
		while (token != NULL)
		{
			dependson[i] = strtol(token, NULL, 10);
			i++;
		}
		ndependson = i;
	}

	else if (!strncmp(s, "PrefetchTag: ", sizeof("PrefetchTag: ")))
	{
		s = s + sizeof("PrefecthTag: ");
		submitorder = strtol(s, NULL, 10);
		sched_type += 1;
	}

	else if (!strncmp(s, "Parameters: ", sizeof("Parameters: ")))
	{
		s = s + sizeof("Parameters: ");
		char * delim = " ";
		char * token = strtok(s, delim);
		int i = 0;
		 
		while (token != NULL)
		{
			parameters[i] = strtol(token, NULL, 10);
			i++;
		}
		nparam = i;
	}

	else if (!strncmp(s, "MemoryNode: ", sizeof("MemoryNode: ")))
	{
		s = s + sizeof("MemoryNode: ");
		memnode = strtol(s, NULL, 10);
	}
	
	else if (!strncmp(s, "Workerorder: ", sizeof("Workerorder: ")))
	{
		s = s + sizeof("Workerorder: ");
		workerorder = strtol(s, NULL, 10);
	}
}


void recordSchedInfo(FILE * f)
{
	size_t lnsize = 128;
	char * s = malloc(sizeof(*s) * lnsize);
	
	while(!feof(f))
	{
		fgets(s, lnsize, f); /* Get the line */
		while(!strcmp(s, "\n")) /* As long as the line is not only a newline symbol (emptyline) do {...} */
		{	  
			checkField(s);
		}

		struct task * task;
			
		HASH_FIND(hh, sched_data, &submitorder, sizeof(submitorder), task);
		
		if (sched_type == 1) /* Only 2 conditions are possible (== 1 or == 0) */
		{
			if (task == NULL)
			{
				_STARPU_MALLOC(task, sizeof(*task));
				task->s_type = sched_type;
				task->submitorder = submitorder;
				CPY(dependson, task->dependson, ndependson);
				task->ndependson = ndependson;
				task->pref_dep = -1;
				
				add_to_hash = 1;
			}

			else
			{
				task->s_type += sched_type;
				CPY(dependson, task->dependson, ndependson);
				task->ndependson = ndependson;
				task->pref_dep = -1;
			}
			
			starpu_task_init(&task->tasks[0]);
			task->tasks[0].workerorder = workerorder;
			task->tasks[0].priority = priority;
			task->tasks[0].workerids = workers;
			task->tasks[0].workerids_len = nworkers;

			unsigned i;
			for(i = 0; i < ndependson ; i++)
			{
				/* Create false task as dependences (they are added later) */
				struct task * taskdep;
				HASH_FIND(hh, sched_data, &dependson[i], sizeof(dependson[i]), taskdep);

				if (taskdep == NULL)
				{
					_STARPU_MALLOC(taskdep, sizeof(*taskdep));
					starpu_task_init(&taskdep->tasks[1]);
					taskdep->submitorder = dependson[i];
					taskdep->tasks[1].cl = NULL;
					taskdep->tasks[1].destroy = 0;
					taskdep->tasks[1]. no_submitorder = 1;

					HASH_ADD(hh, sched_data, submitorder, sizeof(submitorder), taskdep);
				}
			}

			if (add_to_ash)
				HASH_ADD(hh, sched_data, submitorder, sizeof(submitorder), task)
		}

		else
		{
			if (task == NULL)
			{
				_STARPU_MALLOC(task, sizeof(*task));
				task->s_type = sched_type;
				task->submitorder = submitorder;


				add_to_hash = 1;
			}

			else
			{
				task->s_type += shced_type;
				
			}

			task->pref_dep = dependson[0];
			
			struct task * deptask;
			HASH_FIND(hh, sched_data, &task->pref_dep, sizeof(task->pref_dep), deptask);

			if (deptask == NULL)
			{
				_STARPU_MALLOC(deptask, sizeof(*deptask));
				deptask->submitorder = task->pref_dep;
			}

			deptask->send = 1;
			deptask->nparameters = nparam;
			CPY(parameters, deptask->parameters, nparam);
			
			starpu_task_create(task->pref_task);
			deptask->pref_task.cl_prefetch;
			deptask->pref_task.no_submitorder = 1;
			deptask->pref_task.destroy = 1;

			HASH_ADD(hh, sched_data, task->pref_dep, sizeof(task->pref_dep), deptask);

		      				
			task->memory_node = memnode;			

			if (add_to_ash)
				HASH_ADD(hh, sched_data, submitorder, sizeof(submitorder), task)
			
		}

		/* reset some values */
		sched_type = -1;
		add_to_hash = 0;
		
	}
}


void parsing(FILE * f)
{
	recordSchedInfo(f);
}

void put_info(struct starpu_task * task, unsigned submit_order)
{
	struct task * tmptask;
	HASH_FIND(hh, sched_data, &submit_order, sizeof(submitorder), tmptask);

       	if (tmptask == NULL)
		return;

	if (tmptask->s_type == 2 || tmptask->s_type == 1)
	{
		task->workerorder = tmptask->tasks[0].workerorder;
		task->priority = tmptask->tasks[0].priority;
		task->workerids_len = tmptask->tasks[0].workerids_len;
		CPY(tmptask->tasks[0].workerids, task->workerids, task->workerids_len);

		struct starpu_task * deps[tmptask->ndependson];

		unsigned i;
		for(i = 0; i < tmptask->ndependson ; i++)
		{
			struct task * taskdep;
			HASH_FIND(hh, sched_data, &tmptask->dependson[i], sizeof(tmptask->dependson[i]), taskdep);

			if (taskdep == NULL)
			{
				fprintf(stderr, "Can not find the dependence of task(submitorder: %d) according the sched.rec\n", submit_order);
				exit(EXIT_FAILURE);
			}

			deps[i] = &taskdep->tasks[1];
		}

		/* According to the StarPU documentation, these dependences will be added
		   to other existing dependences for this task */
		
		starpu_task_declare_deps_array(task, tmptask->ndependson, deps);
	}

	if (tmptask->s_type == 0 || tmptask->s_type == 2)
	{
		int ret = starpu_task_submit(&tmptask->tasks[1]);
		if (ret != 0)
		{
			fprintf(stderr, "Unable to submit a the false task (corresponding to a false task of the task with the submitorder: %d)", submit_order);
		}
	}

	if(tmptask->pref_dep != -1) /* If the task has a dependence for prefetch */
	{
		struct task * receive_data;
		HASH_FIND(sched_data, &submit_order, sizeof(submit_order), receive_data);
		/* TODO : mettre le handle de receive_data->task dans task.handles */
	}

	if (tmptask->send) /* If the task has stored data to be prefetched */
	{
		struct task * send_data;
		HASH_FIND(hh, sched_data, &submit_order, sizeof(submit_order), send_data);

		if(send_data == NULL)
		{
			fprintf(stderr, "Unable to send_data data for prefetch (submitorder: %d)", submit_order);
			exit(EXIT_FAILURE);
		}
		CPY(&task.handles, send_data->pref_task.handles[0], send_data)
		send_data->pref_task.handles[0] = task->handles[0];
		send_data->pref_task.callback_arg = &tmptask->memory_node;
		/* NOTE : Do it need a function in .callback_func ? */
	}
}

FILE * schedRecInit(const char * filename)
{
	FILE * f = fopen(filename, "r");
	
	if(f == NULL)
	{
		return NULL;
	}
	parsing(f);
	
	return f;
}

void applySchedRec(struct starpu_task * task, unsigned submit_order)
{
	put_info(task, submit_order);
	return;
}
