#include <datawizard/footprint.h>

void compute_buffers_footprint(job_t j)
{
	uint32_t footprint = 0;
	unsigned buffer;

	for (buffer = 0; buffer < j->nbuffers; buffer++)
	{
		data_state *state = j->buffers[buffer].state;

		STARPU_ASSERT(state->ops);
		STARPU_ASSERT(state->ops->footprint);

		footprint = state->ops->footprint(state, footprint);
	}

	j->footprint = footprint;
	j->footprint_is_computed = 1;
}

inline uint32_t compute_data_footprint(data_state *state)
{
	uint32_t interfaceid = state->ops->interfaceid;

	return state->ops->footprint(state, interfaceid);
}
