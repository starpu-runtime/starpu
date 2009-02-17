#ifndef __DATA_CONCURRENCY_H__
#define __DATA_CONCURRENCY_H__

#include <core/jobs.h>

#ifdef NO_DATA_RW_LOCK

unsigned submit_job_enforce_data_deps(job_t j);

void notify_data_dependencies(data_state *data);

#endif

#endif // __DATA_CONCURRENCY_H__

