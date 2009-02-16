#include <core/dependencies/data-concurrency.h>

#ifdef NO_DATA_RW_LOCK

/* When a new task is submitted, we make sure that there cannot be codelets
   with concurrent data-access at the same time in the scheduling engine (eg.
   there can be 2 tasks reading a piece of data, but there cannot be one
   reading and another writing) */
unsigned submit_job_enforce_data_deps(job_t j)
{
	/* TODO */
	return 0;
}

#endif
