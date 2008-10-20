#include <datawizard/footprint.h>

void compute_buffers_footprint(job_t j)
{
	uint32_t footprint = 0;
	unsigned buffer;

	for (buffer = 0; buffer < j->nbuffers; buffer++)
	{
		data_state *state = j->buffers[buffer].state;

		ASSERT(state->ops);
		ASSERT(state->ops->footprint);

		footprint = state->ops->footprint(state, footprint);
	}

	j->footprint = footprint;
}
