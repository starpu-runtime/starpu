#include <sched_ctx_hypervisor_intern.h>
#include <sched_ctx_hypervisor.h>
#include <common/utils.h>
#include <pthread.h>

struct starpu_sched_ctx_hypervisor_criteria* criteria = NULL;
pthread_mutex_t resize_mutex;

static void reset_ctx_wrapper_info(unsigned sched_ctx)
{
	hypervisor.resize_granularity = 1;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		hypervisor.sched_ctx_wrapper[sched_ctx].priority[i] = 0;
	 	hypervisor.sched_ctx_wrapper[sched_ctx].current_idle_time[i] = 0.0;
		hypervisor.sched_ctx_wrapper[sched_ctx].current_working_time[i] = 0.0;
	}
}

struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(void)
{
	hypervisor.resize = 1;
	hypervisor.num_ctxs = 0;
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		hypervisor.sched_ctx_wrapper[i].sched_ctx = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_wrapper[i].current_nprocs = 0;
		hypervisor.sched_ctx_wrapper[i].min_nprocs = 0;
		hypervisor.sched_ctx_wrapper[i].max_nprocs = 0;
		hypervisor.sched_ctx_wrapper[i].current_nprocs = 0;
		reset_ctx_wrapper_info(i);
		int j;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
		{
			hypervisor.sched_ctx_wrapper[i].max_idle_time[j] = MAX_IDLE_TIME;
			hypervisor.sched_ctx_wrapper[i].min_working_time[j] = MIN_WORKING_TIME;
	
		}
	}
	PTHREAD_MUTEX_INIT(&resize_mutex, NULL);

	criteria = (struct starpu_sched_ctx_hypervisor_criteria*)malloc(sizeof(struct starpu_sched_ctx_hypervisor_criteria));
	criteria->update_current_idle_time =  sched_ctx_hypervisor_update_current_idle_time;
	return criteria;
}

void sched_ctx_hypervisor_shutdown(void)
{
	hypervisor.resize = 0;
	PTHREAD_MUTEX_DESTROY(&resize_mutex);
	free(criteria);
}

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx)
{
	hypervisor.sched_ctx_wrapper[sched_ctx].sched_ctx = sched_ctx;
	hypervisor.num_ctxs++;
}

void sched_ctx_hypervisor_ignore_ctx(unsigned sched_ctx)
{
	hypervisor.sched_ctx_wrapper[sched_ctx].sched_ctx = STARPU_NMAX_SCHED_CTXS;
	hypervisor.num_ctxs--;
}

void sched_ctx_hypervisor_set_resize_interval(unsigned sched_ctx, unsigned min_nprocs, unsigned max_nprocs)
{
	hypervisor.sched_ctx_wrapper[sched_ctx].min_nprocs = min_nprocs;
	hypervisor.sched_ctx_wrapper[sched_ctx].max_nprocs = max_nprocs;
}

void sched_ctx_hypervisor_set_resize_granularity(unsigned sched_ctx, unsigned granularity)
{
	hypervisor.resize_granularity = granularity;
}

void sched_ctx_hypervisor_set_idle_max_value(unsigned sched_ctx, int max_idle_value, int *workers, int nworkers)
{
	int i;
	for(i = 0; i < nworkers; i++)
		hypervisor.sched_ctx_wrapper[sched_ctx].max_idle_time[workers[i]] = max_idle_value;
}

void sched_ctx_hypervisor_set_work_min_value(unsigned sched_ctx, int min_working_value, int *workers, int nworkers)
{
	int i;
	for(i = 0; i < nworkers; i++)
		hypervisor.sched_ctx_wrapper[sched_ctx].min_working_time[workers[i]] = min_working_value;

}

void sched_ctx_hypervisor_increase_priority(unsigned sched_ctx, int priority_step, int *workers, int nworkers)
{
	int i;
	for(i = 0; i < nworkers; i++)
		hypervisor.sched_ctx_wrapper[sched_ctx].priority[workers[i]] += priority_step;

}

static int compute_priority_per_sched_ctx(unsigned sched_ctx)
{
	struct sched_ctx_wrapper *sched_ctx_wrapper = &hypervisor.sched_ctx_wrapper[sched_ctx];
	int i;
	int total_priority = 0;

	int nworkers_ctx = starpu_get_nworkers_of_ctx(sched_ctx);
	int *workers = starpu_get_workers_of_ctx(sched_ctx);

	sched_ctx_wrapper->current_nprocs = nworkers_ctx;

	for(i = 0; i < sched_ctx_wrapper->current_nprocs; i++)
		total_priority += sched_ctx_wrapper->priority[workers[i]] + sched_ctx_wrapper->current_idle_time[workers[i]];

	return total_priority;
}

static struct sched_ctx_wrapper* find_highest_priority_sched_ctx(unsigned sched_ctx)
{
	int i;
	int highest_priority = 0;
	int current_priority = 0;
	struct sched_ctx_wrapper *sched_ctx_wrapper = NULL;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		if(hypervisor.sched_ctx_wrapper[i].sched_ctx != STARPU_NMAX_SCHED_CTXS && i != sched_ctx)
		{
			current_priority = compute_priority_per_sched_ctx(i);
			if (highest_priority < current_priority)
			{
				highest_priority = current_priority;
				sched_ctx_wrapper = &hypervisor.sched_ctx_wrapper[i];
			}
		}
	}
	
	return sched_ctx_wrapper;
}

static int* sort_workers_by_priority(unsigned sched_ctx, int worker)
{
	int nworkers_ctx = starpu_get_nworkers_of_ctx(sched_ctx);
	int *workers = starpu_get_workers_of_ctx(sched_ctx);
	
	struct sched_ctx_wrapper *sched_ctx_wrapper = &hypervisor.sched_ctx_wrapper[sched_ctx];
	sched_ctx_wrapper->current_nprocs = nworkers_ctx;

	int curr_workers[nworkers_ctx];

	int k;
	for(k = 0; k < nworkers_ctx; k++)
		curr_workers[k] = workers[k];

	unsigned i;
	int temp;
	for(i = 1; i < nworkers_ctx; i++)
	{
		if(curr_workers[i] == worker)
		{
			temp = curr_workers[0];
			curr_workers[0] = curr_workers[i];
			curr_workers[i] = temp;
		}
		else if(sched_ctx_wrapper->priority[workers[i - 1]] >
			sched_ctx_wrapper->priority[workers[i]])
		{
			temp = curr_workers[i - 1];
			curr_workers[i - 1] = curr_workers[i];
			curr_workers[i] = temp;
		}
	}

	return curr_workers;
}

static void resize_ctxs_if_possible(unsigned sched_ctx, int worker)
{
	struct sched_ctx_wrapper *highest_priority_sched_ctx = find_highest_priority_sched_ctx(sched_ctx);
	struct sched_ctx_wrapper *current_sched_ctx = &hypervisor.sched_ctx_wrapper[sched_ctx];
	if(highest_priority_sched_ctx != NULL && current_sched_ctx->sched_ctx != STARPU_NMAX_SCHED_CTXS)
	{	
		/* sort workers by priority in order to find the first ones with the lowest
		   priority in the current ctx and move them to the ctx with the highest priority*/
		int *ordered_workers = sort_workers_by_priority(sched_ctx, worker);
		unsigned nworkers_to_be_moved = 0;
		
		unsigned potential_nprocs = highest_priority_sched_ctx->current_nprocs +
			hypervisor.resize_granularity;
		
		if(potential_nprocs < highest_priority_sched_ctx->max_nprocs &&
		   potential_nprocs > current_sched_ctx->min_nprocs)
			nworkers_to_be_moved = hypervisor.resize_granularity;
		
		if(nworkers_to_be_moved > 0)
		{
			int workers_to_be_moved[nworkers_to_be_moved];
			
			int i, j = 0;
			for(i = 0; i < current_sched_ctx->current_nprocs; i++)
				workers_to_be_moved[j++] = ordered_workers[i];
			
//			printf("high prio %d\n", highest_priority_sched_ctx->sched_ctx);
//			printf("curr %d\n", current_sched_ctx->sched_ctx);
			
//			printf("n = %d %d\n", nworkers_to_be_moved, workers_to_be_moved[0]);
			starpu_remove_workers_from_sched_ctx(workers_to_be_moved, nworkers_to_be_moved, sched_ctx);
			starpu_add_workers_to_sched_ctx(workers_to_be_moved, nworkers_to_be_moved, highest_priority_sched_ctx->sched_ctx);
			reset_ctx_wrapper_info(sched_ctx);
//			printf("done resize \n");
		}
	}
}

void sched_ctx_hypervisor_update_current_idle_time(unsigned sched_ctx, int worker, double idle_time, unsigned current_nprocs)
{
	struct sched_ctx_wrapper *sched_ctx_wrapper = &hypervisor.sched_ctx_wrapper[sched_ctx];
	sched_ctx_wrapper->current_idle_time[worker] += idle_time;
	if(hypervisor.resize && hypervisor.num_ctxs > 1 &&
	   sched_ctx_wrapper->current_idle_time[worker] > sched_ctx_wrapper->max_idle_time[worker])
	{
		int ret = pthread_mutex_trylock(&resize_mutex);
		if(ret != EBUSY)
		{
			resize_ctxs_if_possible(sched_ctx, worker);
			pthread_mutex_unlock(&resize_mutex);
		}
	}
}

void sched_ctx_hypervisor_update_current_working_time(unsigned sched_ctx, int worker, double working_time, unsigned current_nprocs)
{
	hypervisor.sched_ctx_wrapper[sched_ctx].current_working_time[worker] += working_time;
	hypervisor.sched_ctx_wrapper[sched_ctx].current_nprocs = current_nprocs;
	resize_ctxs_if_possible(sched_ctx, worker);
}

