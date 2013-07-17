#include <starpu_sched_node.h>
#include <starpu_scheduler.h>
#include <float.h>
/* set implementation, task->predicted and task->predicted_transfer with the first worker of workers that can execute that task
 * or have to be calibrated
 */
static void select_best_implementation_and_set_preds(struct starpu_bitmap * workers, struct starpu_task * task)
{
	int best_impl = -1;
	double len = DBL_MAX;
	int workerid;
	for(workerid = starpu_bitmap_first(workers);
	    -1 != workerid;
	    workerid = starpu_bitmap_next(workers, workerid))
	{
		int impl;
		for(impl = 0; impl < STARPU_MAXIMPLEMENTATIONS; impl++)
		{
			if(starpu_worker_can_execute_task(workerid, task, impl))
			{
				enum starpu_perfmodel_archtype archtype = starpu_worker_get_perf_archtype(workerid);
				double d = starpu_task_expected_length(task, archtype, impl);
				if(isnan(d))
				{
					best_impl = impl;
					len = 0.0;
					break;
				}
				if(d < len)
				{
					len = d;
					best_impl = impl;
				}
			}
		}
		if(best_impl != -1)
			break;
	}
	int memory_node = starpu_worker_get_memory_node(workerid);
	task->predicted = len;
	task->predicted_transfer = starpu_task_expected_data_transfer_time(memory_node, task);
	starpu_task_set_implementation(task, best_impl);
}




static int select_best_implementation_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node->nchilds == 1);
	select_best_implementation_and_set_preds(node->workers_in_ctx, task);
	return node->childs[0]->push_task(node->childs[0],task);
}

static struct starpu_task * select_best_implementation_pop_task(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	struct starpu_task * t;
	STARPU_ASSERT(node->is_homogeneous);
	if(!node->fathers[sched_ctx_id])
		return NULL;
	t = node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id], sched_ctx_id);
	if(t)
		select_best_implementation_and_set_preds(node->workers_in_ctx, t);
	return t;
}

struct starpu_sched_node * starpu_sched_node_best_implementation_create(void * ARG STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	node->push_task = select_best_implementation_push_task;
	node->pop_task = select_best_implementation_pop_task;
	return node;
}






