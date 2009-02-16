#ifndef __DATA_CONCURRENCY_H__
#define __DATA_CONCURRENCY_H__

#include <core/jobs.h>

unsigned submit_job_enforce_data_deps(job_t j);

#endif
