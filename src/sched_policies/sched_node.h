#ifndef __SCHED_NODE_H__
#define __SCHED_NODE_H__

void _starpu_sched_node_lock_all_workers(void);
void _starpu_sched_node_unlock_all_workers(void);
void _starpu_sched_node_lock_worker(int workerid);
void _starpu_sched_node_unlock_worker(int workerid);


struct _starpu_worker * _starpu_sched_node_worker_get_worker(struct starpu_sched_node *);
struct _starpu_combined_worker * _starpu_sched_node_combined_worker_get_combined_worker(struct starpu_sched_node * worker_node);

#endif
