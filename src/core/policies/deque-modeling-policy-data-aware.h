#ifndef __DEQUE_MODELING_POLICY_DATA_AWARE_H__
#define __DEQUE_MODELING_POLICY_DATA_AWARE_H__

#include <core/workers.h>
#include <core/mechanisms/queues.h>
#include <core/mechanisms/fifo_queues.h>

void initialize_dmda_policy(struct machine_config_s *config,
 __attribute__ ((unused)) struct sched_policy_s *_policy);

struct jobq_s *get_local_queue_dmda(struct sched_policy_s *policy __attribute__ ((unused)));

#endif // __DEQUE_MODELING_POLICY_DATA_AWARE_H__
