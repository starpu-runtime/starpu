#ifndef __EAGER_CENTRAL_PRIORITY_POLICY_H__
#define __EAGER_CENTRAL_PRIORITY_POLICY_H__

#include <core/workers.h>
#include <core/mechanisms/queues.h>

void initialize_eager_center_priority_policy(struct machine_config_s *config, struct sched_policy_s *policy);
void set_local_queue_eager_priority(struct jobq_s *jobq);
struct jobq_s *get_local_queue_eager_priority(struct sched_policy_s *policy);

#endif // __EAGER_CENTRAL_PRIORITY_POLICY_H__
