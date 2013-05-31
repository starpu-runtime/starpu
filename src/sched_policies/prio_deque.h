#ifndef __PRIO_DEQUE_H__
#define __PRIO_DEQUE_H__
#include <starpu.h>
#include <starpu_task_list.h>


struct _starpu_prio_list
{
	int prio;
	struct starpu_task_list list;
	struct _starpu_prio_list * next;
};

struct _starpu_prio_deque
{
	struct _starpu_prio_list * list;
	unsigned ntasks;
	unsigned nprocessed;
	double exp_start, exp_end, exp_len;
};

void _starpu_prio_deque_init(struct _starpu_prio_deque *);
void _starpu_prio_deque_destroy(struct _starpu_prio_deque *);

int _starpu_prio_deque_is_empty(struct _starpu_prio_deque *);

int _starpu_prio_deque_push_task(struct _starpu_prio_deque *, struct starpu_task*);

struct starpu_task * _starpu_prio_deque_pop_task(struct _starpu_prio_deque*);
struct starpu_task * _starpu_prio_deque_pop_task_for_worker(struct _starpu_prio_deque*, int workerid);

// deque a task of the higher priority available
struct starpu_task * _starpu_prio_deque_deque_task(struct _starpu_prio_deque *);
struct starpu_task * _starpu_prio_deque_deque_task_for_worker(struct _starpu_prio_deque *, int workerid);

#endif // __PRIO_DEQUE_H__
