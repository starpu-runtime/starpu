#ifndef __SCHED_POLICY_H__
#define __SCHED_POLICY_H__

#include <core/mechanisms/queues.h>
//#include <core/mechanisms/work_stealing_queues.h>
//#include <core/mechanisms/central_queues.h>
//#include <core/mechanisms/central_queues_priorities.h>

#include <core/workers.h>

struct machine_config_s;

struct sched_policy_s {
	/* create all the queues */
	void (*init_sched)(struct machine_config_s *, struct sched_policy_s *);

	/* anyone can request which queue it is associated to */
	struct jobq_s *(*get_local_queue)(struct sched_policy_s *);

	/* some worker may block until some activity happens in the machine */
	pthread_cond_t sched_activity_cond;
	pthread_mutex_t sched_activity_mutex;

	pthread_key_t local_queue_key;
};

struct sched_policy_s *get_sched_policy(void);

void init_sched_policy(struct machine_config_s *config);
//void set_local_queue(struct jobq_s *jobq);

int push_task(job_t task);
int push_task_sync(job_t task);
int push_prio_task(job_t task);
struct job_s *pop_task(void);
struct job_s * pop_task_from_queue(struct jobq_s *queue);
struct job_list_s *pop_every_task(void);
struct job_list_s * pop_every_task_from_queue(struct jobq_s *queue);

void wait_on_sched_event(void);

#endif // __SCHED_POLICY_H__
