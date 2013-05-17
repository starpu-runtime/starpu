#include "node_sched.h"
#include <starpu_perfmodel.h>
#include <starpu_scheduler.h>
#include <float.h>


struct _starpu_dmda_data
{
	double alpha;
	double beta;
	double gamma;
	double idle_power;
};



static double compute_fitness_calibration(struct _starpu_sched_node * child,
					  struct _starpu_dmda_data * data STARPU_ATTRIBUTE_UNUSED,
					  struct starpu_task * task STARPU_ATTRIBUTE_UNUSED,
					  struct _starpu_execute_pred *pred)
{
	if(pred->state == CALIBRATING)
		return child->estimated_load(child);
	return DBL_MAX;
}
static double compute_fitness_no_perf_model(struct _starpu_sched_node * child,
					    struct _starpu_dmda_data * data STARPU_ATTRIBUTE_UNUSED,
					    struct starpu_task * task STARPU_ATTRIBUTE_UNUSED,
					    struct _starpu_execute_pred *pred)
{
	if(pred->state == CANNOT_EXECUTE)
		return DBL_MAX;
	return child->estimated_load(child);
}

static double compute_fitness_perf_model(struct _starpu_sched_node * child,
					 struct _starpu_dmda_data * data,
					 struct starpu_task * task,
					 struct _starpu_execute_pred * pred)
{
	if(pred->state == CANNOT_EXECUTE)
		return DBL_MAX;
	return data->alpha * pred->expected_length
		+ data->beta * child->estimated_transfer_length(child, task);
}

static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	struct _starpu_execute_pred preds[node->nchilds];
	int i;
	int calibrating = 0;
	int perf_model = 0;
	int can_execute = 0;
	for(i = 0; i < node->nchilds; i++)
	{
		preds[i] = node->childs[i]->estimated_execute_length(node->childs[i], task);
		switch(preds[i].state)
		{
		case PERF_MODEL:
			perf_model = 1;
			can_execute = 1;
			break;
		case CALIBRATING:
			calibrating = 1;
			can_execute = 1;
			break;
		case NO_PERF_MODEL:
			can_execute = 1;
		case CANNOT_EXECUTE:
			break;
		}
	}
	if(!can_execute)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
		return -ENODEV;
	}
	double (*fitness_fun)(struct _starpu_sched_node *,
			      struct _starpu_dmda_data *,
			      struct starpu_task *,
			      struct _starpu_execute_pred*) = compute_fitness_no_perf_model;
	if(perf_model)
		fitness_fun = compute_fitness_perf_model;
	if(calibrating)
		fitness_fun = compute_fitness_calibration;
	double best_fitness = DBL_MAX;
	int index_best_fitness;
	for(i = 0; i < node->nchilds; i++)
	{
		double tmp = fitness_fun(node->childs[i],
					 node->data,
					 task,
					 preds + i);
		if(tmp < best_fitness)
		{
			best_fitness = tmp;
			index_best_fitness = i;
		}
	}
	struct _starpu_sched_node * c = node->childs[index_best_fitness];

	starpu_task_set_implementation(task, preds[index_best_fitness].impl);
	task->predicted = preds[index_best_fitness].expected_length;
	task->predicted_transfer = c->estimated_transfer_length(c,task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	return c->push_task(c, task);
}


struct _starpu_sched_node * _starpu_sched_node_heft_create(double alpha, double beta, double gamma, double idle_power)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_dmda_data * data = malloc(sizeof(*data));

	data->alpha = alpha;
	data->beta = beta;
	data->gamma = gamma;
	data->idle_power = idle_power;

	node->data = data;
	node->push_task = push_task;
	data->alpha = data->beta = data->gamma = data->idle_power = 0.0;
	//data->total_task_cnt = data->ready_task_cnt = 0;

	return node;
}



