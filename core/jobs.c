#include "jobs.h"

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
	job->type = CODELET;

	job->where = 0;
	job->cb = NULL;
	job->cl = NULL;
	job->argcb = NULL;
	job->synchronous = 0;
	job->use_tag = 0;
	job->nbuffers = 0;
	job->priority = DEFAULT_PRIO;

	job->model = NULL;
	job->predicted = 0.0;

	job->footprint_is_computed = 0;

	job->terminated = 0;

	return job;
}

void handle_job_termination(job_t j)
{
	if (j->terminated)
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
