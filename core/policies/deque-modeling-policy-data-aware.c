#include <core/policies/deque-modeling-policy-data-aware.h>
#include <core/perfmodel/perfmodel.h>

unsigned nworkers;
struct jobq_s *queue_array[NMAXWORKERS];

static job_t dmda_pop_task(struct jobq_s *q)
{
	struct job_s *j;

	j = fifo_pop_task(q);
	if (j) {
		struct fifo_jobq_s *fifo = q->queue;
		double model = j->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = timing_now()/1000000 + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}	

	return j;
}

static void update_data_requests(struct jobq_s *q, struct job_s *j)
{
	uint32_t memory_node = q->memory_node;
	unsigned nbuffers = j->nbuffers;
	unsigned buffer;

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		data_state *state = j->buffers[buffer].state;

		set_data_requested_flag_if_needed(state, memory_node);
	}
}

static double data_penalty(struct jobq_s *q, struct job_s *j)
{
	uint32_t memory_node = q->memory_node;
	unsigned nbuffers = j->nbuffers;
	unsigned buffer;

	double penalty = 0.0;

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		data_state *state = j->buffers[buffer].state;

		if (!is_data_present_or_requested(state, memory_node))
		{
			/* TODO */
			penalty += 100000.0;
		}
	}

	return penalty;
}

static int _dmda_push_task(struct jobq_s *q __attribute__ ((unused)) , job_t task, unsigned prio)
{
	/* find the queue */
	struct fifo_jobq_s *fifo;
	unsigned worker;
	int best = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1;

	double local_task_length[nworkers];
	double local_data_penalty[nworkers];
	double exp_end[nworkers];

	double fitness[nworkers];

	double best_exp_end = 10e240;
	double model_best = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		fifo = queue_array[worker]->queue;

		/* XXX */
		fifo->exp_start = MAX(fifo->exp_start, timing_now()/1000000);
		fifo->exp_end = MAX(fifo->exp_start, timing_now()/1000000);

		if ((queue_array[worker]->who & task->where) == 0)
		{
			/* no one on that queue may execute this task */
			continue;
		}

		local_task_length[worker] = job_expected_length(queue_array[worker]->who,
							task, queue_array[worker]->arch);

		//local_data_penalty[worker] = 0;
		local_data_penalty[worker] = data_penalty(queue_array[worker], task);

		if (local_task_length[worker] == -1.0)
		{
			forced_best = worker;
			break;
		}

		exp_end[worker] = fifo->exp_start + fifo->exp_len + local_task_length[worker];

		if (exp_end[worker] < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end[worker];
		}
	}

	double alpha = 1.0;
	double beta = 1.0;

	double best_fitness = -1;
	
	if (forced_best == -1)
	{
		for (worker = 0; worker < nworkers; worker++)
		{
			fifo = queue_array[worker]->queue;
	
			if ((queue_array[worker]->who & task->where) == 0)
			{
				/* no one on that queue may execute this task */
				continue;
			}
	
			fitness[worker] = alpha*(exp_end[worker] - best_exp_end) 
					+ beta*(local_data_penalty[worker]);

			if (best == -1 || fitness[worker] < best_fitness)
			{
				/* we found a better solution */
				best_fitness = fitness[worker];
				best = worker;

	//			fprintf(stderr, "best fitness (worker %d) %le = alpha*(%le) + beta(%le) \n", worker, best_fitness, exp_end[worker] - best_exp_end, local_data_penalty[worker]);
			}
		}
	}

	STARPU_ASSERT(forced_best != -1 || best != -1);
	
	if (forced_best != -1)
	{
		/* there is no prediction available for that task
		 * with that arch we want to speed-up calibration time
		 * so we force this measurement */
		best = worker;
		model_best = 0.0;
	}
	else 
	{
		model_best = local_task_length[best];
	}

	/* we should now have the best worker in variable "best" */
	fifo = queue_array[best]->queue;

	fifo->exp_end += model_best;
	fifo->exp_len += model_best;

	task->predicted = model_best;

	update_data_requests(queue_array[best], task);

	if (prio) {
		return fifo_push_prio_task(queue_array[best], task);
	} else {
		return fifo_push_task(queue_array[best], task);
	}
}

static int dmda_push_prio_task(struct jobq_s *q, job_t task)
{
	return _dmda_push_task(q, task, 1);
}

static int dmda_push_task(struct jobq_s *q, job_t task)
{
	return _dmda_push_task(q, task, 0);
}

static struct jobq_s *init_dmda_fifo(void)
{
	struct jobq_s *q;

	q = create_fifo();

	q->push_task = dmda_push_task; 
	q->push_prio_task = dmda_push_prio_task; 
	q->pop_task = dmda_pop_task;
	q->who = 0;

	queue_array[nworkers++] = q;

	return q;
}

void initialize_dmda_policy(struct machine_config_s *config, 
 __attribute__ ((unused)) struct sched_policy_s *_policy) 
{
	nworkers = 0;

	setup_queues(init_fifo_queues_mechanisms, init_dmda_fifo, config);
}

struct jobq_s *get_local_queue_dmda(struct sched_policy_s *policy __attribute__ ((unused)))
{
	struct jobq_s *queue;
	queue = pthread_getspecific(policy->local_queue_key);

	if (!queue)
	{
		/* take one randomly as this *must* be for a push anyway XXX */
		queue = queue_array[0];
	}

	return queue;
}

