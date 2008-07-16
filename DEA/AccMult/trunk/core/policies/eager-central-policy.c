#include "sched_policy.h"

/*
 *	This is just the trivial policy where every worker use the same
 *	job queue.
 */

static pthread_key_t local_queue_key;           


