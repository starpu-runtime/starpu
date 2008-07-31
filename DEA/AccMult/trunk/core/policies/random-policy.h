#ifndef __RANDOM_POLICY_H__
#define __RANDOM_POLICY_H__

#include <core/workers.h>
#include <core/mechanisms/queues.h>
#include <core/mechanisms/deque_queues.h>

void initialize_random_policy(struct machine_config_s *config,
 __attribute__ ((unused)) struct sched_policy_s *_policy);

struct jobq_s *get_local_queue_random(struct sched_policy_s *policy __attribute__ ((unused)));

#endif // __RANDOM_POLICY_H__
