#ifndef __DEQUE_MODELING_POLICY_H__
#define __DEQUE_MODELING_POLICY_H__

#include <core/workers.h>
#include <core/mechanisms/queues.h>
#include <core/mechanisms/fifo_queues.h>

void initialize_dm_policy(struct machine_config_s *config,
 __attribute__ ((unused)) struct sched_policy_s *_policy);

struct jobq_s *get_local_queue_dm(struct sched_policy_s *policy __attribute__ ((unused)));

#endif // __DEQUE_MODELING_POLICY_H__
