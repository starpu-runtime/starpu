#ifndef __SCHED_POLICY_H__
#define __SCHED_POLICY_H__

#include <core/mechanisms/queues.h>

struct sched_policy_s {
	/* create all the queues */
	void (*init_sched)(void);

	/* anyone can request which queue it is associated to */
	struct jobq_s *(*get_local_queue)(void);
};

#endif // __SCHED_POLICY_H__
