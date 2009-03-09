#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/data-concurrency.h>

size_t job_get_data_size(job_t j)
{
	size_t size = 0;

	unsigned buffer;
	for (buffer = 0; buffer < j->nbuffers; buffer++)
	{
		data_state *state = j->buffers[buffer].state;
		size += state->ops->get_size(state);
	}

	return size;
}

job_t job_create(void)
{
	job_t job;

	job = job_new();

	job->cb = NULL;
	job->cl = NULL;
	job->argcb = NULL;
	job->synchronous = 0;
	job->use_tag = 0;
	job->nbuffers = 0;
	job->priority = DEFAULT_PRIO;

	job->predicted = 0.0;

	job->footprint_is_computed = 0;

	job->terminated = 0;

	return job;
}

void handle_job_termination(job_t j)
{
	if (UNLIKELY(j->terminated))
		fprintf(stderr, "OOPS ... job %p was already terminated !!\n", j);

	j->terminated = 1;

	/* in case there are dependencies, wake up the proper tasks */
	notify_dependencies(j);

	/* the callback is executed after the dependencies so that we may remove the tag 
 	 * of the task itself */
	if (j->cb)
		j->cb(j->argcb);

	if (j->synchronous)
		sem_post(&j->sync_sem);
}

/* application should submit new tasks to StarPU through this function */
int submit_job(job_t j)
{
	STARPU_ASSERT(j);

	if (!worker_exists(j->cl->where))
		return -ENODEV;

	/* enfore task dependencies */
	if (j->use_tag)
	{
		if (submit_job_enforce_task_deps(j))
			return 0;
	}

#ifdef NO_DATA_RW_LOCK
	/* enforce data dependencies */
	if (submit_job_enforce_data_deps(j))
		return 0;
#endif

	return push_task(j);
}

int submit_prio_job(job_t j)
{
	j->priority = MAX_PRIO;
	
	return submit_job(j);
}

/* note that this call is blocking, and will not make StarPU progress,
 * so it must only be called from the programmer thread, not by StarPU.
 * NB: This also means that it cannot be submitted within a callback ! */
int submit_job_sync(job_t j)
{
	int ret;

	j->synchronous = 1;
	sem_init(&j->sync_sem, 0, 0);

	ret = submit_job(j);
	if (ret == -ENODEV)
	{
		sem_destroy(&j->sync_sem);
		return ret;
	}

	sem_wait(&j->sync_sem);
	sem_destroy(&j->sync_sem);

	return 0;
}
