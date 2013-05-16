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



static void compute_all_things(struct starpu_task * task,
			       struct _starpu_sched_node ** nodes, int nnodes,
			       double * execution_lengths, int * best_impls,//impl used for best execution length, -1 if no execution possible
			       double * transfer_lengths,
			       double * finish_times,
			       int * is_not_calibrated, enum starpu_perf_archtype * arch_not_calibrated, int * impl_not_calibrated,
			       int * is_no_model)
{
	*is_not_calibrated = 0;
	*is_no_model = 1;
	int i = 0;
	for(i = 0; i < nnodes; i++)
	{
		execution_lengths[i] = DBL_MAX;
		best_impls[i] = -1;
		int j;
		for(j = 0; j < STARPU_MAXIMPLEMENTATIONS; j++)
		{
			if(_starpu_sched_node_can_execute_task_with_impl(nodes[i], task, j))
			{
				enum starpu_perf_archtype archtype = starpu_worker_get_perf_archtype(nodes[i]->workerids[0]);
				double d = starpu_task_expected_length(task, archtype, j);
				if(isnan(d))
				{
					*is_not_calibrated = 1;
					*arch_not_calibrated = archtype;
					*impl_not_calibrated = j;
				}
				if(!_STARPU_IS_ZERO(d))//we have a perf model
				{
					*is_no_model = 0;
					if(d < execution_lengths[i])
					{
						execution_lengths[i] = d;
						best_impls[i] = j;
					}
				}
				else//we dont have a perf model for this implementation but we may have one for an other
					if(*is_no_model)
						best_impls[i] = j;
				unsigned memory_node = starpu_worker_get_memory_node(nodes[i]->workerids[0]);
				transfer_lengths[i] = starpu_task_expected_data_transfer_time(memory_node, task);
				finish_times[i] = nodes[i]->estimated_finish_time(nodes[i]);
			}
		}
	}
}

static double compute_total_finish_time(double exp_end, double exp_len, double exp_trans)
{
	if(exp_trans < exp_end)
		return exp_end + exp_len;
	else
		return exp_end + exp_trans;
}

static double fitness(double alpha, double beta, double gamma,
		      double execution_length, double transfer_length, double finish_time, double now)
{
	(void) gamma;
	double total_execution_time = compute_total_finish_time(finish_time - now, execution_length, transfer_length);
	return alpha * total_execution_time + transfer_length * beta;
}

static double fitness_no_model(double alpha, double beta, double transfer_length, double finish_time, double now)
{
	(void) gamma;
	double exp_end = finish_time - now;
	return alpha * exp_end + beta * transfer_length;
}

static double estimated_transfert_time(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node->nworkers);
	unsigned memory_node = starpu_worker_get_memory_node(node->workerids[0]);
	return starpu_task_expected_data_transfer_time(memory_node, task);
}

static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_dmda_data * dt = node->data;
	double execution_lengths[node->nchilds];
	double finish_times[node->nchilds];
	double transfer_lengths[node->nchilds];
	int best_impls[node->nchilds];// -1 mean cant execute
	//double power_consumptions[node->nchilds];
	int i;

	int is_not_calibrated;
	enum starpu_perf_archtype arch_not_calibrated;
	int impl_not_calibrated;

	int is_no_model;

	compute_all_things(task,
			   node->childs, node->nchilds,
			   execution_lengths, best_impls,
			   transfer_lengths,
			   finish_times,
			   &is_not_calibrated, &arch_not_calibrated, &impl_not_calibrated,
			   &is_no_model);

	double max_fitness = DBL_MAX;
	int index_max = -1;
	double now = starpu_timing_now();
	if(is_not_calibrated)
	{
		for(i = 0; i < node->nchilds; i++)
		{
			if(best_impls[i] == -1)
				continue;
			enum starpu_perf_archtype archtype = starpu_worker_get_perf_archtype(node->childs[i]->workerids[0]);
			if(archtype != arch_not_calibrated)
				continue;
			double f = fitness_no_model(dt->alpha, dt->beta, transfer_lengths[i], finish_times[i], now);
			if(f < max_fitness)
			{
				max_fitness = f;
				index_max = i;
			}
		}
	}
	else if(is_no_model)
	{
		for(i = 0; i < node->nchilds; i++)
		{
			if(best_impls[i] == -1)
				continue;
			double f = fitness_no_model(dt->alpha, dt->beta, transfer_lengths[i], finish_times[i], now);
			if(f < max_fitness)
			{
				max_fitness = f;
				index_max = i;
			}
		}
	}
	else
	{
		for(i = 0; i < node->nchilds; i++)
		{
			if(best_impls[i] == -1)
				continue;
			double f =  fitness(dt->alpha, dt->beta, dt->gamma,
					    execution_lengths[i], transfer_lengths[i] , finish_times[i], now);

			if(f < max_fitness)
			{
				max_fitness = f;
				index_max = i;
			}
		}
	}

	STARPU_ASSERT(index_max != -1);
	task->predicted = execution_lengths[index_max];
	task->predicted_transfer = transfer_lengths[index_max];
	starpu_task_set_implementation(task, best_impls[index_max]);
	struct _starpu_sched_node * child = node->childs[index_max];
	return child->push_task(child, task);
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


