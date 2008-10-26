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
	job->use_tag = 0;
	job->nbuffers = 0;
	job->priority = DEFAULT_PRIO;

	job->model = NULL;
	job->predicted = 0.0;

	job->footprint_is_computed = 0;

	return job;
}
