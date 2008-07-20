#ifndef __WORK_STEALING_POLICY_H__
#define __WORK_STEALING_POLICY_H__

#include <core/workers.h>
#include <core/mechanisms/queues.h>

void initialize_ws_policy(struct machine_config_s *config, struct sched_policy_s *policy);
struct jobq_s *get_local_queue_ws(struct sched_policy_s *policy);

#endif // __WORK_STEALING_POLICY_H__
