#include "prio_deque.h"
#include <core/workers.h>


void _starpu_prio_deque_init(struct _starpu_prio_deque * pdeque)
{
	memset(pdeque,0,sizeof(*pdeque));
}
void _starpu_prio_deque_destroy(struct _starpu_prio_deque * pdeque)
{
	while(pdeque->list)
	{
		struct _starpu_prio_list * l = pdeque->list;
		pdeque->list = l->next;
		STARPU_ASSERT(starpu_task_list_empty(&l->list));
		free(l);
	}
}

int _starpu_prio_deque_is_empty(struct _starpu_prio_deque * pdeque)
{
	
	return pdeque->ntasks == 0;
/*
	struct _starpu_prio_list * l = pdeque->list;
	while(l)
	{
		if(!starpu_task_list_empty(&l->list))
			return 0;
		l = l->next;
	}
	return 1;
*/
}


static struct _starpu_prio_list * _starpu_prio_list_create(int prio)
{
	struct _starpu_prio_list * l = malloc(sizeof(*l));
	memset(l, 0, sizeof(*l));
	l->prio = prio;
	return l;
}



int _starpu_prio_deque_push_task(struct _starpu_prio_deque * pdeque, struct starpu_task * task)
{
	STARPU_ASSERT(task != NULL);
        struct _starpu_prio_list * l;
	if(pdeque->list == NULL)
		pdeque->list =  l = _starpu_prio_list_create(task->priority);
	else
	{
		struct _starpu_prio_list * current = pdeque->list;
		struct _starpu_prio_list * prev  = NULL;
		while(current)
		{
			if(current->prio <= task->priority)
				break;
			prev = current;
			current = current->next;
		}
		if(!current)
			prev->next = current = l = _starpu_prio_list_create(task->priority);
		if(current->prio == task->priority)
			l = current;
		if(prev == NULL)
		{
			l = pdeque->list;
			pdeque->list = _starpu_prio_list_create(task->priority);
			pdeque->list->next = l;
			l = pdeque->list;
		}
		else
		{
			l = _starpu_prio_list_create(task->priority);
			l->next = current;
			prev->next = l;
		}
	}
	
	starpu_task_list_push_back(&l->list, task);
	pdeque->ntasks++;
	return 0;
}


static inline int pred_true(struct starpu_task * t STARPU_ATTRIBUTE_UNUSED, void * v STARPU_ATTRIBUTE_UNUSED)
{
	return 1;
}

static inline int pred_can_execute(struct starpu_task * t, void * pworkerid)
{
	int i;
	for(i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if(starpu_worker_can_execute_task(*(int*)pworkerid, t,i))
			return 1;
	return 0;
}


#define REMOVE_TASK(pdeque, first_task_field, next_task_field, predicate, parg)	\
	({								\
		struct _starpu_prio_list * l = pdeque->list;		\
		struct starpu_task * t = NULL;				\
		while(l)						\
		{							\
			t = l->list.first_task_field;			\
			while(t && !predicate(t,parg))			\
				t = t->next_task_field;			\
			if(t)						\
			{						\
				starpu_task_list_erase(&l->list, t);	\
				l = NULL;				\
			}						\
			if(l)						\
				l = l->next;				\
		}							\
		if(t)							\
		{							\
			pdeque->ntasks--;				\
		}							\
		t;							\
	})

struct starpu_task * _starpu_prio_deque_pop_task(struct _starpu_prio_deque * pdeque)
{
	struct starpu_task * t = REMOVE_TASK(pdeque, head, prev, pred_true, STARPU_POISON_PTR);
	return t;
}
struct starpu_task * _starpu_prio_deque_pop_task_for_worker(struct _starpu_prio_deque * pdeque, int workerid)
{
	return REMOVE_TASK(pdeque, head, prev, pred_can_execute, &workerid);
}

// deque a task of the higher priority available
struct starpu_task * _starpu_prio_deque_deque_task(struct _starpu_prio_deque * pdeque)
{
	return REMOVE_TASK(pdeque, tail, next, pred_true, STARPU_POISON_PTR);
}

struct starpu_task * _starpu_prio_deque_deque_task_for_worker(struct _starpu_prio_deque * pdeque, int workerid)
{
	return REMOVE_TASK(pdeque, tail, next, pred_can_execute, &workerid);
}
