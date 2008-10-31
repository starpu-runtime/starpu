#ifndef __NO_PRIO_POLICY_H__
#define __NO_PRIO_POLICY_H__

#include <core/workers.h>
#include <core/mechanisms/fifo_queues.h>

void initialize_no_prio_policy(struct machine_config_s *config, struct sched_policy_s *policy);
//void set_local_queue_eager(struct jobq_s *jobq);
struct jobq_s *get_local_queue_no_prio(struct sched_policy_s *policy);

#endif // __NO_PRIO_POLICY_H__
