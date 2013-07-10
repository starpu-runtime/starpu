#ifndef __SCHED_NODE_H__
#define __SCHED_NODE_H__

void _starpu_sched_node_lock_all_workers(void);
void _starpu_sched_node_unlock_all_workers(void);
void _starpu_sched_node_block_worker(int workerid);
void _starpu_sched_node_unblock_worker(int workerid);

#endif
