#ifndef __FOOTPRINT_H__
#define __FOOTPRINT_H__

#include <core/jobs.h>

struct job_s;

void compute_buffers_footprint(struct job_s *j);
inline uint32_t compute_data_footprint(data_state *state);

#endif // __FOOTPRINT_H__
