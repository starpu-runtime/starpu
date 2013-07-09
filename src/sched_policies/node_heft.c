#include <starpu_sched_node.h>
#include "fifo_queues.h"
#include <starpu_perfmodel.h>
#include <starpu_scheduler.h>
#include <float.h>


struct _starpu_heft_data
{
	double alpha;
	double beta;
	double gamma;
	double idle_power;
	struct starpu_sched_node * no_perf_model_node;
	struct starpu_sched_node * calibrating_node;
};


static double compute_fitness(struct _starpu_heft_data * d, double exp_end, double best_exp_end, double max_exp_end, double transfer_len, double local_power)
{
	return d->alpha * (exp_end - best_exp_end)
		+ d->beta * transfer_len
		+ d->gamma * local_power
		+ d->gamma * d->idle_power * (exp_end - max_exp_end);
}

static int push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_heft_data * d = node->data;	
	struct starpu_sched_node * best_node = NULL;
	double estimated_ends[node->nchilds];
	double estimated_ends_with_task[node->nchilds];
	double best_exp_end_with_task = DBL_MAX;
	double max_exp_end_with_task = 0.0;
	double estimated_lengths[node->nchilds];
	double estimated_transfer_length[node->nchilds];
	int suitable_nodes[node->nchilds];
	int nsuitable_nodes = 0;
	double now = starpu_timing_now();
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		if(starpu_sched_node_execute_preds(c, task, estimated_lengths + i))
		{
			if(isnan(estimated_lengths[i]))
				return d->calibrating_node->push_task(d->calibrating_node, task);
			if(_STARPU_IS_ZERO(estimated_lengths[i]))
				return d->no_perf_model_node->push_task(d->no_perf_model_node, task);
			estimated_transfer_length[i] = starpu_sched_node_transfer_length(c, task);
			estimated_ends[i] = c->estimated_end(c);
			estimated_ends_with_task[i] = starpu_sched_compute_expected_time(now,
											 estimated_ends[i],
											 estimated_lengths[i],
											 estimated_transfer_length[i]);
			if(estimated_ends_with_task[i] < best_exp_end_with_task)	
				best_exp_end_with_task = estimated_ends_with_task[i];
			if(estimated_ends_with_task[i] > max_exp_end_with_task)
				max_exp_end_with_task = estimated_ends_with_task[i];
			suitable_nodes[nsuitable_nodes++] = i;
		}
	}

#if 0
	fprintf(stderr,"estimated end           ");
	for(i = 0; i < node->nchilds; i++)
	{
		fprintf(stderr,"%.0f ",estimated_ends[i]);
	}
	fprintf(stderr,"\n");
	fprintf(stderr,"estimated end with task ");
		for(i = 0; i < node->nchilds; i++)
	{
		fprintf(stderr,"%.0f ",estimated_ends_with_task[i]);
	}
	fprintf(stderr,"\n");
	fprintf(stderr,"transfere length        ");
		for(i = 0; i < node->nchilds; i++)
	{
		fprintf(stderr,"%.0f ",estimated_transfer_length[i]);
	}
	fprintf(stderr,"\n");
	fprintf(stderr,"estimated length        ");
		for(i = 0; i < node->nchilds; i++)
	{
		fprintf(stderr,"%.0f ",estimated_lengths[i]);
	}
	fprintf(stderr,"\n\n");
#endif


	double best_fitness = DBL_MAX;
	int best_inode = -1;
	for(i = 0; i < nsuitable_nodes; i++)
	{
		int inode = suitable_nodes[i];
		double tmp = compute_fitness(d,
					     estimated_ends_with_task[inode],
					     best_exp_end_with_task,
					     max_exp_end_with_task,
					     estimated_transfer_length[inode],
					     0.0);
		if(tmp < best_fitness)
		{
			best_fitness = tmp;
			best_inode = inode;
		}
	}
	//	fprintf(stderr,"push on %d\n",best_inode);
	STARPU_ASSERT(best_inode != -1);
	best_node = node->childs[best_inode];
	return best_node->push_task(best_node, task);
}



#define _STARPU_SCHED_ALPHA_DEFAULT 1.0
#define _STARPU_SCHED_BETA_DEFAULT 1.0
#define _STARPU_SCHED_GAMMA_DEFAULT 1000.0
static double alpha = _STARPU_SCHED_ALPHA_DEFAULT;
static double beta = _STARPU_SCHED_BETA_DEFAULT;
static double _gamma = _STARPU_SCHED_GAMMA_DEFAULT;

#ifdef STARPU_USE_TOP
static const float alpha_minimum=0;
static const float alpha_maximum=10.0;
static const float beta_minimum=0;
static const float beta_maximum=10.0;
static const float gamma_minimum=0;
static const float gamma_maximum=10000.0;
static const float idle_power_minimum=0;
static const float idle_power_maximum=10000.0;
#endif /* !STARPU_USE_TOP */

static double idle_power = 0.0;

#ifdef STARPU_USE_TOP
static void param_modified(struct starpu_top_param* d)
{
#ifdef STARPU_DEVEL
#warning FIXME: get sched ctx to get alpha/beta/gamma/idle values
#endif
	/* Just to show parameter modification. */
	fprintf(stderr,
		"%s has been modified : "
		"alpha=%f|beta=%f|gamma=%f|idle_power=%f !\n",
		d->name, alpha,beta,_gamma, idle_power);
}
#endif /* !STARPU_USE_TOP */

void _heft_add_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	starpu_sched_node_add_child(node, child);
	struct _starpu_heft_data * data = node->data;
	starpu_sched_node_add_child(data->no_perf_model_node,child);
	starpu_sched_node_add_child(data->calibrating_node, child);
}

void _heft_remove_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	starpu_sched_node_remove_child(node, child);
	struct _starpu_heft_data * data = node->data;
	starpu_sched_node_remove_child(data->no_perf_model_node,child);
	starpu_sched_node_remove_child(data->calibrating_node, child);
}

struct starpu_sched_node * starpu_sched_node_heft_create(struct starpu_heft_data * params)
{
	struct starpu_sched_node * node = starpu_sched_node_create();

	struct _starpu_heft_data * data = malloc(sizeof(*data));
	data->alpha = params->alpha;
	data->beta = params->beta;
	data->gamma = params->gamma;
	data->idle_power = params->idle_power;

	data->no_perf_model_node = params->no_perf_model_node_create(params->arg_no_perf_model);
	starpu_bitmap_destroy(data->no_perf_model_node->workers);
	starpu_bitmap_destroy(data->no_perf_model_node->workers_in_ctx);
	data->no_perf_model_node->workers = node->workers;
	data->no_perf_model_node->workers_in_ctx = node->workers_in_ctx;

	data->calibrating_node = params->calibrating_node_create(params->arg_calibrating_node);
	starpu_bitmap_destroy(data->calibrating_node->workers);
	starpu_bitmap_destroy(data->calibrating_node->workers_in_ctx);
	data->calibrating_node->workers = node->workers;
	data->calibrating_node->workers_in_ctx = node->workers_in_ctx;

	node->push_task = push_task;
	node->add_child = _heft_add_child;
	node->remove_child = _heft_remove_child;
	node->data = data;

	return node;
}

int starpu_sched_node_is_heft(struct starpu_sched_node * node)
{
	return node->push_task == push_task;
}



static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		alpha = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		beta = atof(strval_beta);

	const char *strval_gamma = getenv("STARPU_SCHED_GAMMA");
	if (strval_gamma)
		_gamma = atof(strval_gamma);

	const char *strval_idle_power = getenv("STARPU_IDLE_POWER");
	if (strval_idle_power)
		idle_power = atof(strval_idle_power);

#ifdef STARPU_USE_TOP
	starpu_top_register_parameter_float("DMDA_ALPHA", &alpha,
					    alpha_minimum, alpha_maximum, param_modified);
	starpu_top_register_parameter_float("DMDA_BETA", &beta,
					    beta_minimum, beta_maximum, param_modified);
	starpu_top_register_parameter_float("DMDA_GAMMA", &_gamma,
					    gamma_minimum, gamma_maximum, param_modified);
	starpu_top_register_parameter_float("DMDA_IDLE_POWER", &idle_power,
					    idle_power_minimum, idle_power_maximum, param_modified);
#endif /* !STARPU_USE_TOP */

	
	struct starpu_sched_tree * t = starpu_sched_tree_create(sched_ctx_id);
	struct starpu_heft_data data =
		{
			.alpha = alpha,
			.beta = beta,
			.gamma = _gamma,
			.idle_power = idle_power,
			.no_perf_model_node_create = starpu_sched_node_random_create,
			. arg_no_perf_model = NULL,
			.calibrating_node_create = starpu_sched_node_random_create,
			.arg_calibrating_node = NULL
		};
	t->root = starpu_sched_node_heft_create(&data);
	
	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(i);
		STARPU_ASSERT(worker_node);

		struct starpu_sched_node * impl_node = starpu_sched_node_best_implementation_create(NULL);
		impl_node->add_child(impl_node, worker_node);
		starpu_sched_node_set_father(worker_node, impl_node, sched_ctx_id);

		t->root->add_child(t->root, impl_node);
		starpu_sched_node_set_father(impl_node, t->root, sched_ctx_id);
	}

	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(t, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}




struct starpu_sched_policy _starpu_sched_tree_heft_policy =
{
	.init_sched = initialize_heft_center_policy,
	.deinit_sched = deinitialize_heft_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_node_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_node_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "tree-heft",
	.policy_description = "heft tree policy"
};
